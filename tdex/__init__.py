"""Contact-annotated agentic manipulation on GR-1 Tabletop.

Import isolation (enforced by tests/test_isolation.py):
  * `tdex.privileged` (recorder, oracle) is imported only by scripts/.
  * judge, planner, renderer, reader, gate, boundary and recovery never import
    `tdex.sim` or `tdex.privileged`; they see frames and finger summaries only.
"""
