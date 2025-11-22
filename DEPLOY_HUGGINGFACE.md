# 🚀 Deploy to HuggingFace Spaces - Quick Guide

## 📋 Pre-Deployment Checklist

Make sure you have these files ready:

```
alvin-faith/
├── app.py                          # ✅ Entry point (updated for retro arcade)
├── gradio_retro_arcade.py          # ✅ Main demo code
├── requirements_hf.txt             # ✅ Dependencies
├── README.space                    # ✅ Space configuration
├── src/                            # ✅ Source code folder
│   ├── context_aware_agent.py
│   ├── core/
│   │   ├── expanded_temporal_observer.py
│   │   ├── enhanced_snake_game.py
│   │   ├── simple_pacman_game.py
│   │   ├── simple_dungeon_game.py
│   │   └── local_view_game.py
│   └── ...
└── checkpoints/                    # ✅ Pre-trained models
    └── multi_game_enhanced_*_policy.pth
```

## 🎯 Step-by-Step Deployment

### Option 1: GitHub → HuggingFace (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add retro arcade demo for HuggingFace"
   git push origin main
   ```

2. **Create HuggingFace Space**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `retro-arcade-ai` (or your choice)
   - License: MIT
   - SDK: **Gradio**
   - Hardware: CPU Basic (free) or GPU T4 (faster)

3. **Link GitHub Repository**
   - In Space settings, connect your GitHub repo
   - Auto-sync: Enable
   - Branch: main
   - Path: / (root)

4. **Configure Space**
   - Rename `README.space` to `README.md` in Space
   - Or copy contents to Space README
   - Set emoji: 🕹️
   - Set title: "Retro Arcade - Multi-Game AI"

### Option 2: Direct Upload

1. **Create Space** (same as above)

2. **Upload Files via Web UI**
   - Click "Files and versions"
   - Upload these files:
     ```
     app.py
     gradio_retro_arcade.py
     requirements_hf.txt
     ```
   - Create folders and upload:
     ```
     src/ folder → all Python files
     checkpoints/ folder → .pth model files
     ```

3. **Wait for Build**
   - Space will automatically install dependencies
   - First build takes ~2-5 minutes
   - Check build logs for errors

### Option 3: Git CLI

```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/retro-arcade-ai
cd retro-arcade-ai

# Copy files
cp ../alvin-faith/app.py .
cp ../alvin-faith/gradio_retro_arcade.py .
cp ../alvin-faith/requirements_hf.txt .
cp -r ../alvin-faith/src .
cp -r ../alvin-faith/checkpoints .

# Create README from template
cp ../alvin-faith/README.space README.md

# Commit and push
git add .
git commit -m "Initial commit: Retro Arcade AI demo"
git push
```

## 🔧 Configuration Files

### requirements_hf.txt
```
gradio>=4.0.0
numpy>=1.24.0
torch>=2.0.0
Pillow>=9.0.0
```

### README.md (in Space)
Copy contents from `README.space` - this becomes your Space's landing page.

## ⚙️ Space Settings

Recommended settings in HuggingFace Space:

- **SDK**: Gradio
- **Python**: 3.10
- **Hardware**:
  - CPU Basic (free, works fine)
  - CPU Upgrade ($0.50/hr, faster loading)
  - GPU T4 ($0.60/hr, fastest inference)

- **Secrets** (if needed):
  - Add any API keys or secrets here
  - Access via `os.environ['SECRET_NAME']`

- **Environment Variables**:
  - `GRADIO_SERVER_NAME`: 0.0.0.0
  - `GRADIO_SERVER_PORT`: 7860

## 🧪 Testing After Deployment

1. **Wait for Build**: Usually 2-5 minutes
2. **Check Status**: Green checkmark = ready
3. **Test Features**:
   - Load Model button
   - Game selection (all 4 games)
   - Manual controls (arrow buttons)
   - Auto-play buttons
   - Difficulty slider

4. **Common Issues**:

   **Build fails**:
   - Check requirements_hf.txt syntax
   - Verify all imports in gradio_retro_arcade.py
   - Check Python version compatibility

   **Model not loading**:
   - Verify checkpoint file exists
   - Check file path in model_path textbox
   - Try glob pattern: `checkpoints/*.pth`

   **Games not rendering**:
   - Check PIL/Pillow installation
   - Verify game classes are imported
   - Test locally first: `python gradio_retro_arcade.py`

## 📊 Performance Tips

### For Faster Loading:
1. Use smaller checkpoint files (~50MB ideal)
2. Enable GPU hardware (T4)
3. Reduce `cell_size` in renderers (faster rendering)
4. Lower `max_steps` in games

### For Better UX:
1. Add loading spinners
2. Show model loading status clearly
3. Add game instructions
4. Include performance metrics

## 🎮 Customization for HuggingFace

### Change Default Game:
```python
# In create_demo(), line ~477
game_type = gr.Radio(
    choices=["snake", "pacman", "dungeon", "local_view"],
    value="pacman",  # ← Change this
    label="🎮 GAME SELECT"
)
```

### Change Default Model Path:
```python
# In create_demo(), line ~485
model_path = gr.Textbox(
    value="checkpoints/multi_game_enhanced_*_policy.pth",  # ← Update this
    label="🤖 MODEL PATH"
)
```

### Adjust Difficulty:
```python
# In create_demo(), line ~489
difficulty = gr.Slider(0, 2, value=1, step=1,  # ← Default = 1 (Medium)
```

## 📱 Mobile Optimization

For better mobile experience:

1. **Reduce cell_size**: 25 → 20 pixels
2. **Smaller images**: height=650 → 500
3. **Compact controls**: Use icon buttons
4. **Responsive layout**: Test on mobile view

## 🐛 Debugging

### Enable Debug Mode:
```python
# In app.py
demo.launch(debug=True)
```

### Check Logs:
- HuggingFace Space → Settings → Logs
- Look for Python errors
- Check model loading messages

### Test Locally First:
```bash
python gradio_retro_arcade.py
# Visit http://localhost:7860
```

## 🔗 After Deployment

1. **Share URL**:
   - https://huggingface.co/spaces/YOUR_USERNAME/retro-arcade-ai

2. **Embed in Website**:
   ```html
   <iframe src="https://YOUR_USERNAME-retro-arcade-ai.hf.space"
           width="100%" height="800px"></iframe>
   ```

3. **Add to README**:
   - Update main project README with Space link
   - Add badges and demo link

4. **Social Media**:
   - Tweet your Space!
   - Share on Reddit, LinkedIn
   - Add to personal portfolio

## 🎉 Success Indicators

Your Space is working if:
- ✅ Green checkmark on Space page
- ✅ Games render without errors
- ✅ AI makes moves (not random only)
- ✅ Scores update correctly
- ✅ All 4 games work
- ✅ Retro styling displays properly

## 📞 Support

If you encounter issues:

1. Check HuggingFace Status: https://status.huggingface.co
2. HuggingFace Docs: https://huggingface.co/docs/hub/spaces
3. Gradio Docs: https://gradio.app/docs/
4. Community: https://discuss.huggingface.co/

---

**Ready to deploy? Let's go! 🚀**

INSERT COIN TO CONTINUE 🕹️
