wmic Removal Fix — Windows Host Info Fallback
=============================================

**Branch:** ``feat/setup-and-wmic-fix``

Cause
-----

Microsoft removed ``wmic`` (Windows Management Instrumentation Command-line) from
Windows 11 24H2 and later. Full details:

  https://support.microsoft.com/en-us/topic/windows-management-instrumentation-command-line-wmic-removal-from-windows-e9e83c7f-4992-477f-ba1d-96f694b8665d

``get_host_info()`` in ``gprMax/utilities/host_info.py`` used ``wmic`` to query
machine manufacturer, model, and CPU name on Windows. With ``wmic`` absent, the
``subprocess.check_output`` call raises ``FileNotFoundError``. The existing
``except subprocess.CalledProcessError`` did not catch it, so the function crashed
before any simulation could start.

Fix
---

Each of the three ``wmic`` calls (manufacturer, model, CPU) is now wrapped in a
``try/except (subprocess.CalledProcessError, FileNotFoundError)`` block. When either
error is raised, a PowerShell ``Get-CimInstance`` command is used as a fallback to
retrieve the same information:

- Manufacturer → ``Win32_ComputerSystemProduct | Select-Object -ExpandProperty Vendor``
- Model        → ``Win32_ComputerSystem | Select-Object -ExpandProperty Model``
- CPU name     → ``Win32_Processor | Select-Object -ExpandProperty Name``

The CPU loop was also updated to skip empty lines and the wmic ``Name`` header instead
of filtering by ``"CPU" in line``, which failed for modern processor names (e.g.
Intel Core Ultra, AMD Ryzen) that do not contain the word "CPU".

Testing
-------

To verify the fallback is being hit on a wmic absent environment, add a temporary print inside each
``except`` block before the PowerShell call::

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[DEBUG] wmic not found, falling back to PowerShell for manufacturer")
        try:
            ...

Run a simulation::

    python -m gprMax examples/cylinder_Ascan_2D.in

The ``[DEBUG]`` lines confirm the fallback path is active.
