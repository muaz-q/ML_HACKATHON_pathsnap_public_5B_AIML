PathSnap - UI laptop (Meha)
===========================

Muaz's laptop runs the backend (model + API). Your laptop runs only the UI page,
which talks to Muaz's laptop over Wi-Fi. You do NOT need the data or the model.

BEFORE: both laptops on the SAME Wi-Fi. If college Wi-Fi does not work, turn on a
phone hotspot and connect BOTH laptops to it (college Wi-Fi often blocks
laptop-to-laptop connections).

1. Muaz starts start_backend.bat. It prints his address, e.g.
       http://192.168.1.23:8000
   and the API key for the session (if he set one).

2. You double-click check_connection.bat, type only his IP (192.168.1.23).
   It must say [OK]. If it says [FAIL], follow the checklist it prints.

3. You double-click start_ui.bat. The browser opens http://localhost:5500.
   Top bar: enter  http://192.168.1.23:8000  and the API key, press Connect.
   The dot turns green and the 10 trips appear.

4. Click a trip. The route, scores and charts appear. On Muaz's screen a
   REQUEST / RESULT line appears for every click - show that to the evaluator.

Extras: "Run all" scores all trips and shows a summary; arrow keys switch trips;
R or the Play button replays the drive; the buttons on the map switch layers.

If the map background is blank (no internet), the routes still draw - that is fine.
Needs: Windows + any browser. Python is used if installed (recommended); if not,
the page opens straight from the file.
