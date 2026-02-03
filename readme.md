# LLM Code Smell Annotator

Minimal web-based tool to manually annotate LLM-related code smells in GitHub repositories.

The interface loads precomputed repository metadata and source files, displays them side-by-side, and exports annotations as JSON.

No authentication, API keys, or tokens are required.
To enable submission to Google Sheets, set your own Apps Script webhook URL in `manual.html` (`SHEETS_WEBHOOK_URL`).

## Credit : Chenail-Larcher, Zacharie; Mahmoudi, Brahim. 2025.
