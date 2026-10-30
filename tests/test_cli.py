from fileorg.cli import build_parser


def test_cli_parser_does_not_require_organizer_implementation() -> None:
    assert "benchmark" in build_parser().format_help()
