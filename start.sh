#!/bin/bash
# CadenceWorks Analytics Engine — Launcher
# Run this once to install dependencies and start the app

echo ""
echo "================================================"
echo "  CadenceWorks Analytics Engine"
echo "================================================"
echo ""

# Install dependencies if needed
echo "Checking dependencies..."
pip install -r requirements.txt -q

echo ""
echo "Starting app — opening in your browser..."
echo ""

streamlit run app.py
