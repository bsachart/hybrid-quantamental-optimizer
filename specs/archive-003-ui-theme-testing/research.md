# Research: UI Theme Support & Integrated Testing

## Theme Support Strategy

### 1. CSS Variables and Media Queries
The current app hardcodes colors in a `:root` block inside `_inject_styles()`. To support both modes while maintaining the premium look:
- We will define a core set of semantic variables (e.g., `--app-bg`, `--panel-bg`, `--text-main`).
- We will use `@media (prefers-color-scheme: dark)` to override these variables.
- We will also leverage Streamlit's built-in variables (like `--background-color`) to ensure alignment with Streamlit's internal components (like the sidebar or modals).

**Light Mode (Premium Quant):**
- Background: `#f6f1e7` (Beige/Paper)
- Panels: `rgba(255, 255, 251, 0.94)`
- Text: `#1f242a` (Deep Ink)

**Dark Mode (Deep Quant):**
- Background: `#0e1117` (Deep Slate)
- Panels: `rgba(26, 28, 35, 0.94)`
- Text: `#e0e0e0` (Light Gray)

### 2. Altair Chart Theming
Altair charts in `results_display.py` have hardcoded background colors:
```python
.configure(background="#fffdf9")
```
This must be changed to `transparent` or a theme-aware color. Axis labels and grid colors also need to be dynamic.

---

## Streamlit Integration Testing (`AppTest`)

Streamlit's `AppTest` (introduced in 1.28) allows for headless testing of the full app state.
Key interactions to test:
- `file_uploader`: Uploading CSV data.
- `slider`: Setting risk-free rates and target volatility.
- `button`: Triggering the "Solve" logic.
- `session_state`: Verifying that `optimization_complete` becomes true.

**Ousterhout Principle**: Keep the test interface simple. We will create a test helper that prepares mock CSV data to avoid repeating boilerplate.
