# gprMax HTML Helper

Static HTML/JavaScript replacement for the Slint helper.

Open `index.html` in a browser. No build step, server, or package install is required.

The app uses separate tabs for:

- model domain, grid, and time-window commands
- material and dispersion commands
- gprMax geometry commands
- waveform commands
- source commands
- receiver commands
- snapshot commands
- final output

The final `.in` output is read-only and can be copied with the Copy output button. Commands can be removed directly from their command lists; removing a material warns before also removing dependent dispersion and geometry commands. The material tab can create x/y/z material sets for diagonal anisotropy, which are then selected in the geometry tab.
