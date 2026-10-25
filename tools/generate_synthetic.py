#!/usr/bin/env python
"""Generate synthetic English documents (PDF / DOCX / TXT) for the dataset.

Covers the categories that are hard to collect without privacy problems:
invoice, receipt, cv_resume, letter. Other categories need real sources
(see docs/dataset_plan.md).

Design (follows docs/dataset_plan.md)
-------------------------------------
* 5 templates per category, each in 3 "house styles" (wording, currency,
  date format, locale) = 15 groups per category. The group id is written to
  ``data/sources.csv`` as ``template_id`` so ``fileorg split --groups`` keeps
  every group on one side of the split. 15 groups are needed because 5-fold CV
  must still find >= 5 groups on the development side.
* Every document is rendered to a random format (PDF 40 %, DOCX 35 %,
  TXT 25 %), so the extension is not a giveaway.
* About 40 % of the files get neutral names (scan_012.pdf); the rest get
  descriptive names. ``filename_style`` records which is which.
* Fully reproducible: file i of a category depends only on (seed, category, i).
* Re-running removes the files listed as synthetic in sources.csv first, so
  the folder never fills up with stale files.

Usage:  python tools/generate_synthetic.py --per-class 100 --seed 42
Needs:  pip install faker python-docx reportlab
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import escape

BASE_DATE = dt.date(2026, 10, 1)  # fixed, so output does not depend on today's date
DATE_FORMATS = ["%B %d, %Y", "%d %B %Y", "%Y-%m-%d"]
CURRENCIES = ["$", "£", "€"]
LOCALES = ["en_US", "en_GB", "en_US"]
N_VARIANTS = 3
SOURCE_TAG = "synthetic:tools/generate_synthetic.py"

Block = tuple  # ("title", s) ("h", s) ("p", s) ("lines", [s]) ("kv", [(k, v)]) ("table", header, rows, right_cols)


@dataclass
class Ctx:
    rng: random.Random
    fake: object
    v: int  # house-style variant 0..2

    @property
    def sym(self) -> str:
        return CURRENCIES[self.v]

    def L(self, spec: str) -> str:
        """Pick the wording of this house style: L('Invoice no.|Invoice #|Number')."""
        options = spec.split("|")
        return options[self.v % len(options)]

    def date(self, d: dt.date) -> str:
        return d.strftime(DATE_FORMATS[self.v]).replace(" 0", " ")

    def rdate(self, back: int = 900) -> dt.date:
        return BASE_DATE - dt.timedelta(days=self.rng.randrange(back))

    def money(self, x: float) -> str:
        return f"{self.sym}{x:,.2f}"

    def addr(self) -> list[str]:
        return [self.fake.street_address(), f"{self.fake.city()}, {self.fake.postcode()}"]


def _iban(rng: random.Random) -> str:
    return rng.choice(["GB", "DE", "HU", "NL", "FR"]) + f"{rng.randint(10, 99)} " + " ".join(
        f"{rng.randint(0, 9999):04d}" for _ in range(4))


def _ref(rng: random.Random, prefix: str) -> str:
    return f"{prefix}{rng.randint(2022, 2026)}-{rng.randint(1, 9999):04d}"


# ===================================================================== INVOICE
SERVICES = ["Website maintenance", "Consulting services", "Graphic design", "Logo design", "Monthly hosting",
            "Software licence", "Photography session", "Translation services", "Office cleaning",
            "Printing and binding", "Data migration", "Training workshop", "Technical support", "Copywriting"]
GOODS = ["A4 paper (box of 5)", "USB-C cable 2m", "Desk lamp", "Wireless mouse", "Monitor stand", "Toner cartridge",
         "Ergonomic keyboard", "Whiteboard markers", "Ethernet switch 8-port", "Laptop sleeve", "Webcam HD",
         "Standing desk mat"]


def _items(c: Ctx, pool, lo=1, hi=5):
    return [(d, c.rng.randint(1, 10), round(c.rng.uniform(5, 400), 2))
            for d in c.rng.sample(pool, c.rng.randint(lo, hi))]


def _totals(c: Ctx, rows):
    sub = round(sum(q * u for _, q, u in rows), 2)
    rate = c.rng.choice([0, 5, 7, 10, 19, 20, 27])
    tax = round(sub * rate / 100, 2)
    return sub, rate, tax, round(sub + tax, 2)


def _item_table(c: Ctx, rows):
    return ("table", [c.L("Description|Item|Details"), c.L("Qty|Quantity|Units"), c.L("Unit price|Price|Rate"),
                      c.L("Amount|Total|Line total")],
            [[d, str(q), c.money(u), c.money(q * u)] for d, q, u in rows], {1, 2, 3})


def inv_classic(c: Ctx):
    seller, buyer, no = c.fake.company(), c.fake.company(), _ref(c.rng, "INV-")
    d = c.rdate()
    due = d + dt.timedelta(days=c.rng.choice([14, 30, 45]))
    rows = _items(c, SERVICES + GOODS)
    sub, rate, tax, total = _totals(c, rows)
    return [
        ("title", c.L("INVOICE|Invoice|SALES INVOICE")),
        ("lines", [seller, *c.addr(), c.fake.company_email()]),
        ("kv", [(c.L("Invoice number|Invoice No.|Document no."), no), (c.L("Issue date|Date|Invoice date"), c.date(d)),
                (c.L("Due date|Payment due|Pay by"), c.date(due))]),
        ("h", c.L("Bill to|Customer|Invoiced to")),
        ("lines", [buyer, *c.addr()]),
        _item_table(c, rows),
        ("kv", [(c.L("Subtotal|Net total|Sub-total"), c.money(sub)), (f"{c.L('VAT|Tax|Sales tax')} ({rate}%)", c.money(tax)),
                (c.L("Total due|Amount due|Balance due"), c.money(total))]),
        ("p", c.L("Payment terms: net 30 days. Please transfer the total due to the bank account below and quote the "
                  "invoice number.|Payment is due by the date shown above. Late payments may incur interest.|"
                  "Please settle this invoice within the agreed payment terms.")),
        ("kv", [("Bank", f"{c.fake.last_name()} Bank"), ("IBAN", _iban(c.rng))]),
    ], {"ref": no, "who": seller}


def inv_freelance(c: Ctx):
    me, client, no = c.fake.name(), c.fake.company(), f"{c.rng.randint(1, 400):03d}"
    d = c.rdate()
    rows = [(t, c.rng.randint(2, 30), c.rng.choice([35, 45, 60, 75, 90])) for t in
            c.rng.sample(SERVICES, c.rng.randint(1, 4))]
    total = sum(h * r for _, h, r in rows)
    return [
        ("title", c.L("Invoice|INVOICE|Invoice for services")),
        ("lines", [me, *c.addr(), c.fake.email()]),
        ("kv", [(c.L("Invoice #|Invoice no.|Number"), no), ("Date", c.date(d)),
                (c.L("Payable by|Due date|Payment due"), c.date(d + dt.timedelta(days=14)))]),
        ("h", c.L("Client|Bill to|Invoiced to")),
        ("lines", [client, *c.addr()]),
        ("p", f"Invoice for services rendered in {d.strftime('%B %Y')}."),
        ("table", ["Task", "Hours", "Rate", "Amount"],
         [[t, str(h), c.money(r), c.money(h * r)] for t, h, r in rows], {1, 2, 3}),
        ("kv", [(c.L("Total due|Amount due|Total payable"), c.money(total))]),
        ("p", f"Payment is due within 14 days of the invoice date. Please pay by bank transfer or PayPal ({c.fake.email()})."),
    ], {"ref": f"INV-{no}", "who": me}


def inv_tax(c: Ctx):
    seller, buyer, no = c.fake.company(), c.fake.company(), _ref(c.rng, "TI-")
    d = c.rdate()
    rows = _items(c, GOODS + SERVICES, 2, 5)
    rate = c.rng.choice([7, 10, 19, 20])
    table_rows, net, gross = [], 0.0, 0.0
    for desc, q, u in rows:
        n = round(q * u, 2)
        g = round(n * (1 + rate / 100), 2)
        net, gross = net + n, gross + g
        table_rows.append([desc, str(q), c.money(n), f"{rate}%", c.money(g)])
    return [
        ("title", c.L("Tax Invoice|TAX INVOICE|VAT invoice")),
        ("kv", [(c.L("Supplier|Seller|Issued by"), seller), (c.L("Supplier tax ID|VAT ID|Tax number"), f"{c.rng.randint(10**9, 10**10 - 1)}"),
                (c.L("Customer|Buyer|Bill to"), buyer), (c.L("Invoice number|Invoice No.|Ref."), no),
                (c.L("Date of issue|Invoice date|Date"), c.date(d)), (c.L("Payment due|Due date|Pay by"), c.date(d + dt.timedelta(days=30)))]),
        ("table", ["Item", "Qty", "Net", "VAT %", "Gross"], table_rows, {1, 2, 3, 4}),
        ("kv", [("Total net", c.money(net)), ("Total VAT", c.money(gross - net)),
                (c.L("Amount due|Total due|Balance due"), c.money(gross))]),
        ("p", "Payment method: bank transfer. Please use the invoice number as the payment reference."),
    ], {"ref": no, "who": seller}


def inv_utility(c: Ctx):
    company, who = c.fake.company() + c.rng.choice([" Energy", " Telecom", " Water", " Broadband"]), c.fake.name()
    d = c.rdate()
    start = d - dt.timedelta(days=30)
    base, usage, fee = round(c.rng.uniform(8, 40), 2), round(c.rng.uniform(10, 150), 2), round(c.rng.uniform(1, 9), 2)
    acc = f"{c.rng.randint(10**7, 10**8 - 1)}"
    return [
        ("title", c.L("Invoice|INVOICE|Your invoice")),
        ("lines", [company, "Customer service: " + c.fake.phone_number()]),
        ("kv", [(c.L("Account number|Customer no.|Account no."), acc), (c.L("Invoice date|Date|Issued"), c.date(d)),
                ("Billing period", f"{c.date(start)} to {c.date(d)}")]),
        ("lines", [who, *c.addr()]),
        ("table", ["Charge", "Amount"], [["Standing charge", c.money(base)], ["Usage charges", c.money(usage)],
                                          ["Service fee", c.money(fee)]], {1}),
        ("kv", [(c.L("Total amount due|Amount due|Please pay"), c.money(base + usage + fee)),
                (c.L("Please pay by|Due date|Payment due"), c.date(d + dt.timedelta(days=21)))]),
        ("p", "Pay online or by bank transfer using your account number as the reference. A late payment fee may be added to overdue invoices."),
    ], {"ref": f"INV-{acc}", "who": company}


def inv_paid(c: Ctx):
    seller, buyer, no = c.fake.company(), c.fake.name(), _ref(c.rng, "INV-")
    d = c.rdate()
    rows = _items(c, SERVICES + GOODS, 1, 4)
    sub, rate, tax, total = _totals(c, rows)
    return [
        ("title", c.L("INVOICE|Invoice|Invoice (paid)")),
        ("lines", [seller, *c.addr()]),
        ("kv", [(c.L("Invoice number|Invoice No.|Invoice #"), no), (c.L("Invoice date|Date|Issued"), c.date(d)),
                (c.L("Due date|Payment due|Pay by"), c.date(d + dt.timedelta(days=30))),
                ("Status", f"PAID in full on {c.date(d + dt.timedelta(days=c.rng.randint(1, 25)))}")]),
        ("h", c.L("Bill to|Customer|Invoiced to")),
        ("lines", [buyer, *c.addr()]),
        _item_table(c, rows),
        ("kv", [(f"{c.L('VAT|Tax|Sales tax')} ({rate}%)", c.money(tax)), (c.L("Total|Invoice total|Total amount"), c.money(total))]),
        ("p", "This invoice has been settled. Thank you for your business."),
    ], {"ref": no, "who": seller}


# ===================================================================== RECEIPT
RETAIL = [("Milk 1L", 1.29), ("Bread loaf", 2.20), ("Eggs (12)", 3.40), ("Coffee beans 500g", 8.99), ("Bananas 1kg", 1.79),
          ("Pasta 500g", 1.49), ("Olive oil 750ml", 6.50), ("Notebook A5", 2.99), ("Batteries AA (4)", 4.99),
          ("Shampoo", 3.89), ("Orange juice", 2.49), ("Cheddar 250g", 3.20)]
DISHES = [("Caesar salad", 9.50), ("Margherita pizza", 11.00), ("Grilled salmon", 18.50), ("Beef burger", 13.90),
          ("Mushroom risotto", 14.20), ("Soup of the day", 6.50), ("Cheesecake", 6.80), ("Sparkling water", 3.00),
          ("House red wine", 6.50), ("Espresso", 2.80)]
PAYMENTS = ["VISA **** {n}", "MASTERCARD **** {n}", "Cash", "Debit card ending {n}"]


def _pay(c: Ctx) -> str:
    return c.rng.choice(PAYMENTS).format(n=c.rng.randint(1000, 9999))


def rec_pos(c: Ctx):
    store, d = c.fake.company(), c.rdate()
    picks = [(n, c.rng.randint(1, 3), p) for n, p in c.rng.sample(RETAIL, c.rng.randint(2, 7))]
    sub = round(sum(q * p for _, q, p in picks), 2)
    tax = round(sub * c.rng.choice([0.05, 0.08, 0.2]), 2)
    total = round(sub + tax, 2)
    pay, no = _pay(c), f"{c.rng.randint(10**5, 10**6 - 1)}"
    blocks = [
        ("title", store), ("lines", [*c.addr(), "Tel " + c.fake.phone_number()]),
        ("kv", [(c.L("Receipt #|Receipt no.|Transaction"), no),
                ("Date", f"{c.date(d)} {c.rng.randint(8, 21):02d}:{c.rng.randint(0, 59):02d}"),
                ("Cashier", c.fake.first_name())]),
        ("table", ["Item", "Qty", "Price"], [[n, str(q), c.money(q * p)] for n, q, p in picks], {1, 2}),
        ("kv", [("Subtotal", c.money(sub)), ("Tax", c.money(tax)), ("TOTAL", c.money(total)), ("Paid by", pay)]),
    ]
    if pay == "Cash":
        paid = float(int(total) + c.rng.randint(1, 20))
        blocks.append(("kv", [("Cash tendered", c.money(paid)), ("Change", c.money(paid - total))]))
    blocks.append(("p", c.L("THANK YOU FOR YOUR PURCHASE!|Thank you for shopping with us.|Thanks! See you again soon.")))
    blocks.append(("p", "Please keep this receipt for returns within 30 days."))
    return blocks, {"ref": no, "who": store}


def rec_online(c: Ctx):
    who, shop, d = c.fake.name(), c.fake.company(), c.rdate()
    rows = _items(c, GOODS, 1, 3)
    total = round(sum(q * u for _, q, u in rows), 2)
    no = f"{c.rng.randint(10**7, 10**8 - 1)}"
    return [
        ("title", c.L("Payment receipt|Receipt for your payment|Payment confirmation")),
        ("p", f"Thank you for your payment, {who}. We have received your payment of {c.money(total)} to {shop}."),
        ("kv", [("Order number", no), ("Payment date", c.date(d)), ("Payment method", _pay(c)),
                (c.L("Amount paid|Total paid|Paid"), c.money(total))]),
        ("table", ["Item", "Qty", "Amount"], [[n, str(q), c.money(q * u)] for n, q, u in rows], {1, 2}),
        ("p", "This receipt confirms that your payment has been processed. No further action is required."),
    ], {"ref": no, "who": shop}


def rec_restaurant(c: Ctx):
    place, d = c.fake.last_name() + c.rng.choice(["'s Kitchen", " Bistro", " Grill", " Trattoria"]), c.rdate()
    picks = [(n, c.rng.randint(1, 3), p) for n, p in c.rng.sample(DISHES, c.rng.randint(2, 6))]
    sub = round(sum(q * p for _, q, p in picks), 2)
    tip = round(sub * c.rng.choice([0.0, 0.1, 0.12, 0.15]), 2)
    return [
        ("title", place), ("lines", c.addr()),
        ("kv", [("Table", str(c.rng.randint(1, 30))), ("Server", c.fake.first_name()), ("Guests", str(c.rng.randint(1, 6))),
                ("Date", c.date(d))]),
        ("table", ["Dish", "Qty", "Price"], [[n, str(q), c.money(q * p)] for n, q, p in picks], {1, 2}),
        ("kv", [("Subtotal", c.money(sub)), (c.L("Tip|Service charge|Gratuity"), c.money(tip)),
                ("Total paid", c.money(sub + tip)), ("Payment", _pay(c))]),
        ("p", c.L("Thank you for dining with us!|Thank you for your visit.|We hope to see you again.")),
    ], {"ref": f"T{c.rng.randint(100, 999)}", "who": place}


def rec_received(c: Ctx):
    payer, taker, d = c.fake.name(), c.fake.name(), c.rdate()
    purpose = c.rng.choice([f"rent for {d.strftime('%B %Y')}", "annual membership fee", "deposit for the room booking",
                            "tutoring lessons", "sale of a used bicycle", "course registration fee"])
    amount = c.rng.choice([50, 80, 120, 150, 250, 400, 650, 900])
    no = f"R-{c.rng.randint(1, 9999):04d}"
    return [
        ("title", c.L("RECEIPT|Receipt|Official receipt")),
        ("kv", [(c.L("Receipt no.|Receipt #|Number"), no), ("Date", c.date(d))]),
        ("p", f"Received from {payer} the sum of {c.money(amount)} as payment for {purpose}."),
        ("kv", [("Payment method", c.rng.choice(["Cash", "Bank transfer", "Cheque"])), ("Received by", taker)]),
        ("p", "Signature: ______________________"),
    ], {"ref": no, "who": payer}


def rec_subscription(c: Ctx):
    vendor, who, d = c.fake.company(), c.fake.name(), c.rdate()
    plan = c.rng.choice(["Pro plan (monthly)", "Team plan (annual)", "Storage 200 GB", "Premium membership", "Basic plan"])
    amount = c.rng.choice([4.99, 9.99, 12.00, 19.00, 49.00, 99.00])
    no = f"{c.rng.randint(1000, 9999)}-{c.rng.randint(1000, 9999)}"
    return [
        ("title", f"{c.L('Receipt from|Your receipt from|Payment receipt -')} {vendor}"),
        ("kv", [("Receipt number", no), ("Date paid", c.date(d)), ("Payment method", _pay(c)), ("Billed to", who)]),
        ("table", ["Description", "Qty", "Amount"], [[plan, "1", c.money(amount)]], {1, 2}),
        ("kv", [(c.L("Amount paid|Total paid|Paid"), c.money(amount))]),
        ("p", f"If you have any questions about this payment, contact support@{vendor.split()[0].lower().strip(',.')}.com."),
    ], {"ref": no, "who": vendor}


# ===================================================================== CV / RESUME
TITLES = ["Software Engineer", "Marketing Manager", "Accountant", "Data Analyst", "Project Manager", "Graphic Designer",
          "Sales Representative", "HR Specialist", "Mechanical Engineer", "Registered Nurse", "Teacher",
          "Customer Support Agent", "Logistics Coordinator", "Business Analyst"]
BULLETS = ["Managed a team of {n} people and coordinated weekly planning", "Increased sales by {p}% within one year",
           "Prepared monthly financial and performance reports", "Developed and maintained internal tools used by {n} colleagues",
           "Coordinated projects across {n} departments", "Reduced processing time by {p}% through process improvements",
           "Trained and mentored new colleagues", "Handled customer enquiries and escalations",
           "Organised events with up to {n} participants", "Introduced a new reporting system and documented procedures"]
SKILLS = ["Microsoft Excel", "Python", "SQL", "Project management", "Public speaking", "Team leadership", "Customer service",
          "Data analysis", "Adobe Photoshop", "Negotiation", "Time management", "JavaScript", "Budgeting", "Technical writing",
          "Scrum", "Accounting", "Research", "Teamwork"]
DEGREES = ["BSc Computer Science", "MSc Economics", "BA English Literature", "BEng Mechanical Engineering", "MBA",
           "BSc Business Administration", "MA Education", "BSc Nursing", "BA Graphic Design", "MSc Data Science"]
LANGS = ["English (native)", "German (B2)", "Spanish (B1)", "Hungarian (C1)", "French (A2)", "English (fluent)",
         "Italian (B1)", "Polish (B2)"]


def _bullet(c: Ctx) -> str:
    return c.rng.choice(BULLETS).format(n=c.rng.randint(3, 40), p=c.rng.randint(5, 45))


def _lc(s: str) -> str:
    return s[0].lower() + s[1:]


def _bullets(c: Ctx, k: int) -> list[str]:
    return [b.format(n=c.rng.randint(3, 40), p=c.rng.randint(5, 45)) for b in c.rng.sample(BULLETS, k)]


def _jobs(c: Ctx, n: int):
    year, jobs = c.rng.randint(2012, 2019), []
    for _ in range(n):
        end = min(year + c.rng.randint(1, 4), 2026)
        jobs.append((c.rng.choice(TITLES), c.fake.company(), f"{year} – {end}",
                     _bullets(c, c.rng.randint(2, 4))))
        year = end
    return jobs


def _edu(c: Ctx):
    y = c.rng.randint(2005, 2022)
    return (c.rng.choice(DEGREES), f"University of {c.fake.city()}", f"{y - 3} – {y}")


def _contact(c: Ctx, name: str):
    return [c.fake.email(), c.fake.phone_number(), c.fake.city()]


def cv_classic(c: Ctx):
    name = c.fake.name()
    blocks = [("title", name), ("lines", _contact(c, name)),
              ("h", c.L("Profile|Summary|Personal statement")),
              ("p", f"{c.rng.choice(TITLES)} with {c.rng.randint(2, 15)} years of experience. Reliable, organised and motivated to deliver results.")]
    blocks.append(("h", c.L("Work Experience|Professional Experience|Employment History")))
    for title, company, dates, bullets in _jobs(c, c.rng.randint(2, 4)):
        blocks += [("p", f"{title}, {company} ({dates})"), ("lines", [f"- {b}" for b in bullets])]
    deg, uni, dates = _edu(c)
    blocks += [("h", c.L("Education|Education and Qualifications|Academic Background")), ("p", f"{deg}, {uni} ({dates})"),
               ("h", c.L("Skills|Key Skills|Core Competencies")), ("p", ", ".join(c.rng.sample(SKILLS, 6))),
               ("h", c.L("Languages|Language Skills|Languages spoken")), ("p", ", ".join(c.rng.sample(LANGS, 2)))]
    return blocks, {"ref": "", "who": name}


def cv_tech(c: Ctx):
    name, handle = c.fake.name(), c.fake.user_name()
    blocks = [("title", name), ("lines", [c.fake.email(), f"github.com/{handle}", f"linkedin.com/in/{handle}"]),
              ("h", c.L("Technical Skills|Skills|Technologies")),
              ("kv", [("Languages", ", ".join(c.rng.sample(["Python", "Java", "C#", "JavaScript", "TypeScript", "Go", "SQL", "Rust"], 4))),
                      ("Tools", ", ".join(c.rng.sample(["Git", "Docker", "Linux", "Jenkins", "Terraform", "PostgreSQL", "Redis"], 4)))]),
              ("h", "Projects")]
    for _ in range(2):
        blocks.append(("p", f"{c.fake.word().capitalize()}{c.rng.choice(['Hub', 'Flow', 'Track', 'Kit'])} - "
                            f"{c.rng.choice(['web application', 'command line tool', 'mobile app', 'data pipeline'])} with {c.rng.randint(50, 900)} users."))
    blocks.append(("h", c.L("Experience|Work Experience|Employment")))
    for title, company, dates, bullets in _jobs(c, c.rng.randint(1, 3)):
        blocks += [("p", f"{title} | {company} | {dates}"), ("lines", [f"* {b}" for b in bullets])]
    deg, uni, dates = _edu(c)
    blocks += [("h", "Education"), ("p", f"{deg}, {uni}, {dates}")]
    return blocks, {"ref": "", "who": name}


def cv_academic(c: Ctx):
    name = c.fake.name()
    blocks = [("title", "Curriculum Vitae"),
              ("kv", [("Name", name), ("Address", ", ".join(c.addr())), ("Email", c.fake.email())]),
              ("h", c.L("Research Interests|Areas of Research|Research Focus")),
              ("p", ", ".join(c.rng.sample(["machine learning", "urban planning", "organic chemistry", "labour economics",
                                             "medieval history", "network security", "cognitive psychology", "renewable energy"], 3))),
              ("h", "Education")]
    for _ in range(2):
        deg, uni, dates = _edu(c)
        blocks.append(("p", f"{deg}, {uni}, {dates}"))
    blocks.append(("h", c.L("Publications|Selected Publications|Peer-reviewed Publications")))
    blocks.append(("lines", [f"{c.fake.last_name()}, {c.fake.first_name()[0]}. ({c.rng.randint(2012, 2025)}). "
                             f"{c.fake.catch_phrase()}. Journal of {c.rng.choice(['Applied Science', 'Social Studies', 'Computing', 'Economics'])}."
                             for _ in range(c.rng.randint(3, 5))]))
    blocks += [("h", "Teaching Experience"), ("p", f"Lecturer, {c.rng.choice(['Introduction to Statistics', 'Research Methods', 'Academic Writing'])}, {c.rng.randint(2015, 2025)}"),
               ("h", c.L("Awards and Grants|Honours|Awards")), ("p", f"{c.fake.last_name()} Foundation research grant, {c.rng.randint(2014, 2025)}")]
    return blocks, {"ref": "", "who": name}


def cv_entry(c: Ctx):
    name = c.fake.name()
    deg, uni, dates = _edu(c)
    blocks = [("title", name), ("lines", _contact(c, name)),
              ("h", c.L("Objective|Career objective|About me")),
              ("p", f"Recent graduate seeking an entry-level position as a {c.rng.choice(TITLES)} where I can apply my skills and keep learning."),
              ("h", "Education"), ("p", f"{deg}, {uni} ({dates})"),
              ("h", c.L("Internships|Work Experience|Practical Experience")),
              ("p", f"Intern, {c.fake.company()} (summer {c.rng.randint(2020, 2025)})"), ("lines", [f"- {b}" for b in _bullets(c, 2)]),
              ("h", "Skills"), ("p", ", ".join(c.rng.sample(SKILLS, 5))),
              ("h", c.L("Certifications|Courses|Training")), ("p", c.rng.choice(["First Aid certificate", "Google Analytics certification", "Cambridge C1 Advanced", "Scrum Fundamentals"])),
              ("h", "Languages"), ("p", ", ".join(c.rng.sample(LANGS, 2))), ("p", "References available upon request.")]
    return blocks, {"ref": "", "who": name}


def cv_europass(c: Ctx):
    name = c.fake.name()
    blocks = [("title", "Curriculum Vitae"), ("h", "Personal information"),
              ("kv", [("Surname / First name", f"{name.split()[-1]} {name.split()[0]}"), ("Address", ", ".join(c.addr())),
                      ("Telephone", c.fake.phone_number()), ("E-mail", c.fake.email())]),
              ("h", "Work experience"),
              ("table", ["Dates", "Occupation and employer", "Main activities"],
               [[d, f"{t}, {co}", "; ".join(b[:55] for b in bl[:2])] for t, co, d, bl in _jobs(c, c.rng.randint(2, 3))], set()),
              ("h", "Education and training")]
    deg, uni, dates = _edu(c)
    blocks += [("table", ["Dates", "Qualification", "Institution"], [[dates, deg, uni]], set()),
               ("h", "Personal skills"),
               ("kv", [("Mother tongue(s)", c.rng.choice(["English", "Hungarian", "German", "Polish"])),
                       ("Other language(s)", ", ".join(c.rng.sample(LANGS, 2))),
                       ("Digital skills", ", ".join(c.rng.sample(SKILLS, 3)))])]
    return blocks, {"ref": "", "who": name}


# ===================================================================== LETTER
def _closing(c: Ctx, named: bool) -> str:
    if c.v == 0:
        return "Sincerely,"
    return "Yours sincerely," if named else "Yours faithfully,"


def _frame(c: Ctx, sender, recipient, salutation, subject, paras, named, signer, when=None):
    blocks = [("lines", sender), ("p", c.date(when or c.rdate())), ("lines", recipient), ("p", salutation)]
    if subject:
        blocks.append(("p", f"{c.L('Re:|Subject:|Regarding:')} {subject}"))
    blocks += [("p", p) for p in paras]
    blocks += [("p", _closing(c, named)), ("p", signer)]
    return blocks


def let_complaint(c: Ctx):
    me, company = c.fake.name(), c.fake.company()
    d1 = c.rdate()
    ref = f"{c.rng.randint(100000, 999999)}"
    product = c.rng.choice(["a laptop bag", "a washing machine", "a pair of running shoes", "an office chair",
                            "a set of kitchen knives", "a phone case"])
    issue, detail = c.rng.choice([("arrived damaged", "The packaging was torn and the item was visibly scratched."),
                                  ("was delivered more than three weeks late", "I had to rearrange several appointments because of the delay."),
                                  ("does not work as described", "It stopped working after two days of normal use.")])
    remedy = c.rng.choice(["refund the full purchase price", "send a replacement", "credit my account"])
    paras = [f"I am writing to complain about {product} that I ordered from your company on {c.date(d1)} (order reference {ref}). Unfortunately, the item {issue}.",
             f"{detail} I contacted your customer service team on {c.date(d1 + dt.timedelta(days=9))}, but I have not received a satisfactory reply.",
             f"I would be grateful if you would {remedy} within 14 days of receiving this letter. Otherwise I will have to contact a consumer protection organisation.",
             "I look forward to hearing from you."]
    return _frame(c, [me, *c.addr()], [company, *c.addr()], "Dear Sir or Madam,", f"Complaint about order {ref}", paras,
                  False, me, when=d1 + dt.timedelta(days=c.rng.randint(14, 30))), {"ref": ref, "who": me}


def let_cover(c: Ctx):
    me, company, role = c.fake.name(), c.fake.company(), c.rng.choice(TITLES)
    hm = c.fake.last_name()
    named = c.rng.random() < 0.6
    paras = [f"I am writing to apply for the position of {role} at {company}, as advertised on your website. I hold a {c.rng.choice(DEGREES)} and have {c.rng.randint(1, 12)} years of relevant experience.",
             f"In my current role at {c.fake.company()}, I {_lc(_bullet(c))}. I am confident that this experience would allow me to contribute to your team from the first day.",
             "I have enclosed my CV for your consideration and would welcome the opportunity to discuss my application in an interview. Thank you for your time."]
    return _frame(c, [me, *c.addr(), c.fake.email()], [f"Mr/Ms {hm}" if named else "Human Resources", company, *c.addr()],
                  f"Dear Ms {hm}," if named else "Dear Hiring Manager,", f"Application for {role}", paras, named, me), {"ref": "", "who": me}


def let_recommendation(c: Ctx):
    me, person, org = c.fake.name(), c.fake.name(), c.fake.company()
    y1 = c.rng.randint(2012, 2022)
    paras = [f"I am pleased to write this letter of recommendation for {person}, who worked under my supervision as {c.rng.choice(TITLES)} at {org} from {y1} to {y1 + c.rng.randint(1, 4)}.",
             f"During this time {person.split()[0]} proved to be reliable, hard-working and an excellent communicator. {person.split()[0]} {_lc(_bullet(c))}.",
             f"I recommend {person.split()[0]} without reservation. Please feel free to contact me if you require any further information."]
    return _frame(c, [org, *c.addr()], [], "To whom it may concern,", "", paras, False, f"{me}\n{org}"), {"ref": "", "who": person}


def let_notice(c: Ctx):
    company, d = c.fake.company() + " Property Management", c.rdate()
    kind = c.rng.choice(["rent", "water", "meeting"])
    if kind == "rent":
        subject = "Notice of rent adjustment"
        paras = [f"We are writing to inform you that, in line with your tenancy agreement, the monthly rent will be adjusted by {c.rng.randint(2, 7)}% with effect from {c.date(d + dt.timedelta(days=60))}.",
                 "If you have any questions, please contact our office during business hours."]
    elif kind == "water":
        subject = "Planned interruption of water supply"
        paras = [f"Please be advised that essential maintenance work will take place on {c.date(d + dt.timedelta(days=10))}. The water supply will be interrupted between 9:00 and 15:00.",
                 "We apologise for the inconvenience and thank you for your understanding."]
    else:
        subject = "Annual residents' meeting"
        paras = [f"The annual residents' meeting will be held on {c.date(d + dt.timedelta(days=21))} at 18:00 in the community room.",
                 "Your attendance is important, as the budget for the coming year will be discussed."]
    return _frame(c, [company, *c.addr()], [c.fake.name(), *c.addr()], "Dear Resident,", subject, paras, False,
                  f"{c.fake.name()}\nProperty Manager"), {"ref": "", "who": company}


def let_inquiry(c: Ctx):
    me, mine, supplier = c.fake.name(), c.fake.company(), c.fake.company()
    named = c.rng.random() < 0.5
    last = c.fake.last_name()
    what = c.rng.choice(["office furniture", "printing services", "catering for corporate events", "industrial cleaning supplies", "IT hardware"])
    paras = [f"I am writing on behalf of {mine} to enquire about your {what}. We are planning a larger order for the coming quarter.",
             "Could you please send us a quotation, including delivery times and payment terms? We would also appreciate information about volume discounts.",
             "We look forward to your reply and to a possible cooperation."]
    return _frame(c, [mine, *c.addr()], [supplier, *c.addr()], f"Dear Mr {last}," if named else "Dear Sir or Madam,",
                  f"Enquiry about {what}", paras, named, f"{me}\n{mine}"), {"ref": "", "who": me}


GENERATORS: dict[str, list[Callable]] = {
    "invoice": [inv_classic, inv_freelance, inv_tax, inv_utility, inv_paid],
    "receipt": [rec_pos, rec_online, rec_restaurant, rec_received, rec_subscription],
    "cv_resume": [cv_classic, cv_tech, cv_academic, cv_entry, cv_europass],
    "letter": [let_complaint, let_cover, let_recommendation, let_notice, let_inquiry],
}


# ===================================================================== RENDERING
def render_txt(blocks: list[Block]) -> str:
    out: list[str] = []
    for b in blocks:
        kind = b[0]
        if kind in ("title", "h", "p"):
            out.append(b[1])
        elif kind == "lines":
            out.extend(b[1])
        elif kind == "kv":
            out.extend(f"{k}: {v}" for k, v in b[1])
        elif kind == "table":
            header, rows, right = b[1], b[2], b[3]
            widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
            fmt = lambda r: "  ".join(str(x).rjust(w) if i in right else str(x).ljust(w)
                                      for i, (x, w) in enumerate(zip(r, widths))).rstrip()
            out += [fmt(header), "-" * len(fmt(header)), *[fmt(r) for r in rows]]
        out.append("")
    return "\n".join(out).strip() + "\n"


def write_docx(blocks, path: Path, author: str, title: str, created: dt.datetime) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    for b in blocks:
        kind = b[0]
        if kind == "title":
            doc.add_heading(b[1], 0)
        elif kind == "h":
            doc.add_heading(b[1], 2)
        elif kind == "p":
            doc.add_paragraph(b[1])
        elif kind == "lines":
            p = doc.add_paragraph()
            for i, line in enumerate(b[1]):
                run = p.add_run(line)
                if i < len(b[1]) - 1:
                    run.add_break()
        elif kind == "kv":
            for k, v in b[1]:
                p = doc.add_paragraph()
                p.add_run(f"{k}: ").bold = True
                p.add_run(str(v))
        elif kind == "table":
            header, rows, right = b[1], b[2], b[3]
            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Table Grid"
            for i, h in enumerate(header):
                table.rows[0].cells[i].text = h
            for r in rows:
                cells = table.add_row().cells
                for i, x in enumerate(r):
                    cells[i].text = str(x)
                    if i in right:
                        cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    props = doc.core_properties
    props.author, props.last_modified_by, props.title = author, author, title
    props.created = props.modified = created
    doc.save(str(path))


def write_pdf(blocks, path: Path, font: str, author: str, title: str, pagesize: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    bold = {"Helvetica": "Helvetica-Bold", "Times-Roman": "Times-Bold", "Courier": "Courier-Bold"}[font]
    size = letter if pagesize == "letter" else A4
    margin = 20 * mm
    avail = size[0] - 2 * margin
    body = ParagraphStyle("body", fontName=font, fontSize=10, leading=13)
    cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=11)
    cell_r = ParagraphStyle("cellr", parent=cell, alignment=TA_RIGHT)
    head = ParagraphStyle("head", parent=body, fontName=bold, fontSize=12, leading=15, spaceBefore=8, spaceAfter=3)
    ttl = ParagraphStyle("title", parent=body, fontName=bold, fontSize=20, leading=24, spaceAfter=8)

    def esc(s) -> str:
        return escape(str(s).encode("cp1252", "replace").decode("cp1252")).replace("\n", "<br/>")

    story = []
    for b in blocks:
        kind = b[0]
        if kind == "title":
            story.append(Paragraph(esc(b[1]), ttl))
        elif kind == "h":
            story.append(Paragraph(esc(b[1]), head))
        elif kind == "p":
            story += [Paragraph(esc(b[1]), body), Spacer(1, 5)]
        elif kind == "lines":
            story += [Paragraph("<br/>".join(esc(x) for x in b[1]), body), Spacer(1, 6)]
        elif kind == "kv":
            data = [[Paragraph(f"<b>{esc(k)}</b>", cell), Paragraph(esc(v), cell)] for k, v in b[1]]
            t = Table(data, colWidths=[avail * 0.32, avail * 0.68], hAlign="LEFT")
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [t, Spacer(1, 6)]
        elif kind == "table":
            header, rows, right = b[1], b[2], b[3]
            lens = [min(max(len(str(r[i])) for r in [header] + rows), 50) + 4 for i in range(len(header))]
            widths = [avail * n / sum(lens) for n in lens]
            data = [[Paragraph(f"<b>{esc(h)}</b>", cell_r if i in right else cell) for i, h in enumerate(header)]]
            data += [[Paragraph(esc(x), cell_r if i in right else cell) for i, x in enumerate(r)] for r in rows]
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [t, Spacer(1, 8)]
    SimpleDocTemplate(str(path), pagesize=size, leftMargin=margin, rightMargin=margin, topMargin=margin,
                      bottomMargin=margin, title=title, author=author).build(story)


# ===================================================================== MAIN
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:40] or "doc"


def _filename(c: Ctx, cat: str, meta: dict, idx: int, ext: str, informative: bool, taken: set[str]) -> str:
    if informative:
        base = {"invoice": f"Invoice_{meta['ref']}", "receipt": f"Receipt_{meta['ref']}",
                "cv_resume": c.rng.choice([f"CV_{meta['who']}", f"Resume_{meta['who']}", "my_cv"]),
                "letter": c.rng.choice([f"Letter_{meta['who']}", "letter", "cover_letter", "formal_letter"])}[cat]
    else:
        base = f"{c.rng.choice(['scan', 'document', 'file', 'untitled', 'new_doc', 'doc'])}_{idx:03d}"
    name, n = f"{_slug(base)}.{ext}", 2
    while name in taken:
        name = f"{_slug(base)}_{n}.{ext}"
        n += 1
    taken.add(name)
    return name


def _load_sources(path: Path):
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw", help="dataset root (one sub-folder per category)")
    ap.add_argument("--sources", default="data/sources.csv")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--categories", nargs="*", default=list(GENERATORS), choices=list(GENERATORS))
    args = ap.parse_args(argv)

    from faker import Faker
    fakers = {loc: Faker(loc) for loc in set(LOCALES)}
    out, src_path = Path(args.out), Path(args.sources)

    # remove the files of the previous synthetic run, keep real entries
    fields, rows = _load_sources(src_path)
    kept = []
    for row in rows:
        if (row.get("source") or "").startswith("synthetic:"):
            (out / row["file"]).unlink(missing_ok=True)
        else:
            kept.append(row)

    new_rows, seen_hashes, counts = [], set(), {}
    today = dt.date.today().isoformat()
    for cat in args.categories:
        folder = out / cat
        folder.mkdir(parents=True, exist_ok=True)
        templates, taken = GENERATORS[cat], {p.name for p in folder.iterdir()}
        n_groups = len(templates) * N_VARIANTS
        for i in range(args.per_class):
            group = i % n_groups
            t_index, variant = group // N_VARIANTS, group % N_VARIANTS
            for attempt in range(20):  # regenerate on an exact duplicate
                rng = random.Random(f"{args.seed}/{cat}/{i}/{attempt}")
                fake = fakers[LOCALES[variant]]
                fake.seed_instance(rng.randrange(2 ** 31))
                c = Ctx(rng, fake, variant)
                blocks, meta = templates[t_index](c)
                digest = hashlib.sha1(render_txt(blocks).encode()).hexdigest()
                if digest not in seen_hashes:
                    seen_hashes.add(digest)
                    break
            ext = rng.choices(["pdf", "docx", "txt"], [0.40, 0.35, 0.25])[0]
            informative = rng.random() < 0.6
            name = _filename(c, cat, meta, i, ext, informative, taken)
            path = folder / name
            title = next((b[1] for b in blocks if b[0] == "title"), "")
            author = fake.name()
            if ext == "txt":
                path.write_text(render_txt(blocks), encoding="utf-8")
            elif ext == "docx":
                created = dt.datetime.combine(c.rdate(), dt.time(rng.randint(8, 18), rng.randint(0, 59)))
                write_docx(blocks, path, author, title if rng.random() < 0.5 else "", created)
            else:
                write_pdf(blocks, path, rng.choice(["Helvetica", "Times-Roman", "Courier"]), author,
                          title if rng.random() < 0.5 else "", "letter" if variant == 0 else "a4")
            counts[(cat, ext)] = counts.get((cat, ext), 0) + 1
            new_rows.append({"file": f"{cat}/{name}", "source": f"{SOURCE_TAG} seed={args.seed}", "licence": "CC0 (generated)",
                             "date": today, "template_id": f"{cat}_t{t_index + 1}v{variant + 1}",
                             "filename_style": "informative" if informative else "neutral"})

    fieldnames = list(dict.fromkeys(fields + ["file", "source", "licence", "date", "template_id", "filename_style"]))
    src_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(kept + new_rows)

    print(f"Generated {len(new_rows)} files in {out}; sources written to {src_path}")
    for cat in args.categories:
        print(f"  {cat:<10}", {e: counts.get((cat, e), 0) for e in ("pdf", "docx", "txt")})
    return 0


if __name__ == "__main__":
    sys.exit(main())