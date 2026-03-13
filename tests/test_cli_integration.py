import runpy


def test_package_main_module_invokes_agent_main(monkeypatch):
    called = []

    monkeypatch.setattr("nodifyctx.agent.main", lambda: called.append("package_main"))

    runpy.run_module("nodifyctx.__main__", run_name="__main__")

    assert called == ["package_main"]