from ci.model_quality.placeholders import resolve_argument, resolve_argv


def test_json_and_regex_braces_are_preserved():
    argument = (
        '{"name":"exact_match,flexible-extract",'
        '"direction":"higher","pattern":"a{2,4}"}'
    )

    assert resolve_argument(argument, {"output_dir": "/output"}) == argument


def test_only_controlled_placeholders_are_replaced():
    argv = [
        "--output",
        "{reports_dir}/result.json",
        "--json",
        '{"name":"score","unknown":"{user_value}"}',
    ]

    assert resolve_argv(argv, {"reports_dir": "/reports"}) == [
        "--output",
        "/reports/result.json",
        "--json",
        '{"name":"score","unknown":"{user_value}"}',
    ]
