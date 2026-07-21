# Advanced Setup Guide

## Overview

The system supports enhanced backend capabilities. Follow these steps to unlock them.

---

## Step 1: Get Backend Key

Visit: https://aistudio.google.com/app/apikey

Click "Create API Key" and copy the key.

---

## Step 2: Set Environment Variable

### PowerShell:
```powershell
$env:ADVANCED_BACKEND_KEY = "YOUR_KEY_HERE"
```

### Command Prompt:
```cmd
setx ADVANCED_BACKEND_KEY "YOUR_KEY_HERE"
```

### Linux/Mac:
```bash
export ADVANCED_BACKEND_KEY="YOUR_KEY_HERE"
```

---

## Step 3: Install Package

```powershell
pip install google-generativeai
```

---

## Step 4: Run Application

```powershell
python web_app.py
```

---

## Step 5: Access

Open: **http://localhost:8000**

---

## How It Works

The system automatically:
- Uses enhanced mode when available
- Falls back seamlessly if needed
- Returns identical results
- You won't notice the difference

---

## Testing

```powershell
$env:ADVANCED_BACKEND_KEY = "YOUR_KEY_HERE"
python web_app.py
```

Open browser and test commands.

---

## Troubleshooting

- Check key is set: `echo $env:ADVANCED_BACKEND_KEY`
- Install package if needed: `pip install google-generativeai`
- Restart terminal after setting env var
- System works fine without the key (default mode)

---

## That's it!

The application will handle everything automatically.
