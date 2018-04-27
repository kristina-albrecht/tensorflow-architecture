#!/bin/bash
pandoc --from=markdown+hard_line_breaks+backtick_code_blocks+abbreviations --top-level-division=chapter --template=./eisvogel.tex --pdf-engine=lualatex --filter=pandoc-crossref --filter=pandoc-citeproc tensorflow-architecture.md -o TensorReport.pdf