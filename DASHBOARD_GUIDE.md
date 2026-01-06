# 🎯 Complete Dashboard Guide

## Getting Started with the Web Dashboard

This guide will walk you through using the Neural Network Dashboard from start to finish.

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Launching the Dashboard](#launching-the-dashboard)
3. [Interface Overview](#interface-overview)
4. [Step-by-Step Tutorial](#step-by-step-tutorial)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Prerequisites

Ensure you have Python 3.8+ installed.

### Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask (web framework)
- PyPDF2 (PDF processing)
- python-docx (DOC/DOCX processing)
- NetworkX, NumPy, etc. (neural network components)

---

## 🚀 Launching the Dashboard

### Option 1: Quick Launch (Recommended)

```bash
python launch_dashboard.py
```

This will:
- ✅ Check and install missing dependencies
- ✅ Create necessary directories
- ✅ Start the server
- ✅ Open your browser automatically

### Option 2: Direct Launch

```bash
python dashboard.py
```

Then navigate to: **http://localhost:5000**

---

## 🖥️ Interface Overview

### Main Components

```
┌─────────────────────────────────────────────────────────┐
│  HEADER                                                 │
│  • Status indicator (🟢 Ready / 🟠 Training / 🔴 Off)  │
│  • System status message                                │
└─────────────────────────────────────────────────────────┘

┌────────────────────────┬────────────────────────────────┐
│  📁 UPLOAD PANEL       │  🎓 TRAINING CONTROL           │
│  • Drag-drop zone      │  • Epochs setting              │
│  • File list           │  • Learning rate               │
│  • Upload button       │  • Training progress           │
└────────────────────────┴────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  💬 QUERY INTERFACE                                     │
│  • Text input box                                       │
│  • Response display                                     │
│  • Module routing info                                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  📊 STATISTICS DASHBOARD                                │
│  • Total Nodes  • Total Edges  • Queries  • Status     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  🕸️ GRAPH VISUALIZATION                                 │
│  • Module tabs (Text / Math / Code)                     │
│  • Interactive D3.js graph                              │
│  • Export buttons                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 Step-by-Step Tutorial

### Tutorial 1: Training on Text Documents

#### Step 1: Upload Files

1. **Prepare your documents**:
   - Text files (.txt)
   - PDF documents
   - Word documents (.doc, .docx)
   - Markdown files (.md)

2. **Upload methods**:
   - **Drag & Drop**: Drag files into the upload zone
   - **Click to Browse**: Click upload zone → select files

3. **Verify upload**:
   - Files appear in the list below
   - File sizes are shown
   - PDF/DOC files show "processed" status

#### Step 2: Configure Training

1. **Set Parameters**:
   - **Epochs**: 5-10 for initial training (more = better, but slower)
   - **Learning Rate**: 0.1 (default is good for most cases)

2. **Click "Start Training"**

3. **Monitor Progress**:
   - Progress bar shows completion percentage
   - Status messages update in real-time
   - Training typically takes 1-5 minutes for 50 files

#### Step 3: Query the System

Once training completes (status shows "Ready - Trained"):

1. **Enter a query** in the text box:
   ```
   Write a short paragraph about artificial intelligence
   ```

2. **Press Enter** or click "Process Query"

3. **View results**:
   - Response appears in the box below
   - Module routing info shows which module handled it
   - Confidence score indicates routing certainty

#### Step 4: Visualize the Graph

1. **Select a module tab** (e.g., "Text Module")

2. **Click "Load Graph"**

3. **Interact with visualization**:
   - **Drag nodes** to reposition
   - **Click nodes** for details
   - **Zoom** with mouse wheel
   - Nodes are colored by domain
   - Edge thickness = connection strength

4. **Export if needed**:
   - JSON for custom processing
   - GraphML for Gephi/Cytoscape
   - DOT for Graphviz

---

### Tutorial 2: Training on Code Files

#### Step 1: Upload Code Files

Supported code files:
- `.py` (Python)
- `.js` (JavaScript)
- `.java` (Java)
- `.cpp`, `.c` (C/C++)
- `.cs` (C#)
- `.rb` (Ruby)
- `.go` (Go)

#### Step 2: Train

Use default settings or:
- **Higher learning rate** (0.15): Faster convergence
- **Fewer epochs** (3-5): Code patterns are clearer

#### Step 3: Query Code Generation

Examples:
```
Create a Python function to calculate factorial
```
```
Write JavaScript code to validate email
```
```
Implement a binary search algorithm
```

The system will generate code templates with documentation!

---

### Tutorial 3: Mixed Training (Text + Code + Math)

#### Upload Mixed Files

- Research papers (PDF)
- Programming tutorials (TXT, MD)
- Code examples (PY, JS)
- Math textbooks (PDF)

#### Benefits

The system automatically:
- ✅ Categorizes content by domain
- ✅ Creates specialized subgraphs
- ✅ Routes queries to appropriate modules

#### Query Examples

| Query | Module | Response Type |
|-------|--------|---------------|
| "Explain quantum computing" | Text | Descriptive text |
| "Calculate 25 * 4 + 10" | Math | Numerical result |
| "Sort function in Python" | Code | Code template |

---

## 🎨 Advanced Features

### 1. Real-Time Training Monitoring

Watch training progress live:
- **Progress bar**: Visual completion indicator
- **Status messages**: Current operation
- **Background training**: Continue using dashboard

### 2. Confidence Scoring

After each query, see:
- **Selected module**: Which handled the query
- **Confidence**: How certain the routing was (0-100%)
- **All scores**: Confidence for each module

Example:
```
Routed to: text_module (Confidence: 100.0%)
All scores: {text_module: 1.0, math_module: 0.0, code_module: 0.0}
```

### 3. Interactive Graph Features

**Node Details**:
- Click any node to see:
  - Label (word/phrase)
  - Frequency (how often it appears)
  - Domain (text/math/code)

**Graph Navigation**:
- Drag nodes to organize
- Zoom in/out for detail
- Force-directed layout auto-organizes

**Export Options**:
- **JSON**: For web applications
- **GraphML**: For Gephi analysis
- **DOT**: For Graphviz rendering

### 4. System Statistics

Monitor performance:
- **Total Nodes**: Across all modules
- **Total Edges**: All connections
- **Query Count**: Queries processed
- **Training Status**: Yes/No

### 5. Module Comparison

View graphs for each module:
- **Text Module**: Natural language patterns
- **Math Module**: Mathematical relationships
- **Code Module**: Programming structures

Compare sizes and structures!

---

## 🔍 Use Cases

### Academic Research

**Scenario**: Analyze 100 research papers

1. Upload PDFs
2. Train with 10 epochs
3. Query: "Summarize key findings about [topic]"
4. Visualize concept relationships

### Code Documentation

**Scenario**: Generate docs from codebase

1. Upload Python files
2. Train with 5 epochs
3. Query: "Explain the [function_name] function"
4. Export code graph for documentation

### Content Generation

**Scenario**: Create writing assistant

1. Upload books/articles in your style
2. Train with 15 epochs
3. Query: "Write in the style of [author]"
4. Generate creative content

### Educational Tool

**Scenario**: Math tutoring system

1. Upload math textbooks (PDF)
2. Train thoroughly (20 epochs)
3. Students query: "Solve x^2 + 5x + 6 = 0"
4. System provides step-by-step solutions

---

## ⚙️ Configuration Tips

### For Small Datasets (<50 files)

```
Epochs: 5
Learning Rate: 0.1
```

### For Medium Datasets (50-200 files)

```
Epochs: 10
Learning Rate: 0.08
```

### For Large Datasets (>200 files)

```
Epochs: 15-20
Learning Rate: 0.05
```

### For Quick Testing

```
Epochs: 3
Learning Rate: 0.15
```

---

## 🐛 Troubleshooting

### Issue: "Template file not found"

**Solution**: Ensure `templates/dashboard.html` exists

```bash
# Check structure
ls templates/dashboard.html
```

### Issue: PDF processing fails

**Cause**: Encrypted or scanned PDFs

**Solutions**:
1. Convert to text first
2. Use OCR for scanned PDFs
3. Remove encryption

### Issue: Training stuck at 0%

**Causes**:
- No files uploaded
- Files in wrong format
- Server error

**Solutions**:
1. Check browser console (F12)
2. Verify files uploaded
3. Restart server

### Issue: Graph won't load

**Causes**:
- Module not trained
- Too many nodes
- JavaScript error

**Solutions**:
1. Train module first
2. Reduce max_nodes in code
3. Check browser console

### Issue: Slow performance

**Solutions**:
1. Reduce file count
2. Lower epochs
3. Close other programs
4. Use smaller graph visualizations

---

## 🎯 Best Practices

### File Organization

Before upload:
```
my_training_data/
├── text/
│   ├── article1.txt
│   ├── paper1.pdf
│   └── ...
├── code/
│   ├── script1.py
│   ├── app.js
│   └── ...
└── mixed/
    └── ...
```

### Training Strategy

1. **Start small**: 10-20 files for testing
2. **Verify results**: Query after initial training
3. **Scale up**: Add more files gradually
4. **Fine-tune**: Adjust epochs/learning rate

### Query Optimization

Good queries:
- ✅ "Write a function to calculate factorial"
- ✅ "Explain quantum entanglement"
- ✅ "Calculate the derivative of x^3"

Poor queries:
- ❌ "code"
- ❌ "help"
- ❌ "what"

### Visualization Tips

1. **Load after training**: Wait for completion
2. **Start with Text module**: Usually largest
3. **Export for analysis**: Use GraphML for deep analysis
4. **Limit nodes**: Use 200-500 for best performance

---

## 📊 Performance Expectations

### Training Speed

| Files | Epochs | Expected Time |
|-------|--------|---------------|
| 10 | 5 | 30 seconds |
| 50 | 5 | 2-3 minutes |
| 100 | 10 | 5-8 minutes |
| 500 | 15 | 20-30 minutes |

### Query Speed

- Simple queries: <100ms
- Complex queries: 100-500ms
- Graph loading: 1-3 seconds

### Resource Usage

- Memory: 50-200MB (depends on data size)
- CPU: Moderate during training
- Disk: Minimal (models ~1-10MB each)

---

## 🎓 Learning Path

### Beginner (Day 1)

1. ✅ Launch dashboard
2. ✅ Upload 10 text files
3. ✅ Train with defaults
4. ✅ Make simple queries
5. ✅ View text module graph

### Intermediate (Week 1)

1. ✅ Mix different file types
2. ✅ Experiment with parameters
3. ✅ Try all modules
4. ✅ Export graphs
5. ✅ Analyze routing confidence

### Advanced (Month 1)

1. ✅ Train on large datasets (>100 files)
2. ✅ Fine-tune hyperparameters
3. ✅ Custom query strategies
4. ✅ Graph analysis in Gephi
5. ✅ Integrate with other tools via API

---

## 🚀 Next Steps

After mastering the dashboard:

1. **Explore API**: Use REST endpoints programmatically
2. **Customize**: Modify dashboard.html for your needs
3. **Scale up**: Train on your full dataset
4. **Integrate**: Connect to other applications
5. **Extend**: Add new modules or features

---

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** - Most common issues covered
2. **Browser console** (F12) - See JavaScript errors
3. **Server logs** - Terminal shows Python errors
4. **Documentation** - Review DASHBOARD_README.md
5. **Test with samples** - Use sample_data/ for verification

---

## 🎉 Success Checklist

Your dashboard is working when you can:

- ✅ Upload files without errors
- ✅ See training progress update
- ✅ Get query responses
- ✅ See routing information
- ✅ Load and interact with graphs
- ✅ Export graphs successfully
- ✅ View system statistics

---

**Congratulations! You're ready to use the Neural Network Dashboard!** 🎊

For technical details, see `DASHBOARD_README.md`  
For API reference, see `PROJECT_DOCUMENTATION.md`  
For command-line usage, see `QUICKSTART.md`

