# 🧠 Neural Network Dashboard

## Web-Based Interface for Graph Neural Network Training and Visualization

A comprehensive web dashboard that provides an intuitive interface for training, querying, and visualizing your modular graph-based neural network system.

---

## 🌟 Features

### 📁 File Upload & Processing
- **Drag-and-drop** file upload interface
- **Multi-format support**: PDF, TXT, DOC, DOCX, MD, and code files (PY, JS, JAVA, CPP, etc.)
- **Automatic text extraction** from PDFs and DOC files
- **Batch processing** of multiple files
- File size limit: 100MB per file

### 🎓 Training Control
- **Configurable training parameters**:
  - Number of epochs (1-50)
  - Learning rate (0.01-1.0)
- **Real-time progress tracking** with progress bar
- **Training statistics** display
- **Background training** - continue using dashboard while training

### 💬 Interactive Query Interface
- **Natural language queries** to all modules
- **Real-time response** generation
- **Module routing information** (shows which module handled the query)
- **Confidence scores** for routing decisions
- Support for:
  - Text generation queries
  - Math calculations
  - Code generation requests

### 🕸️ Graph Visualization
- **Interactive D3.js graphs** with force-directed layout
- **Module selection**: View graphs for Text, Math, or Code modules
- **Color-coded nodes** by domain
- **Weighted edges** (thickness indicates connection strength)
- **Interactive features**:
  - Drag nodes to reposition
  - Click nodes for details
  - Zoom and pan
  - Export graphs (JSON, GraphML, DOT formats)

### 📊 Statistics Dashboard
- **Real-time system status**
- **Module statistics**:
  - Total nodes per module
  - Total edges per module
  - Average edge weights
  - Training status
- **Query counter**
- **Performance metrics**

---

## 🚀 Quick Start

### 1. Start the Dashboard

```bash
python dashboard.py
```

The dashboard will start on `http://localhost:5000`

### 2. Open in Browser

Navigate to: **http://localhost:5000**

### 3. Upload Training Files

- **Drag and drop** files into the upload zone, or
- **Click** the upload zone to browse for files
- Supported formats: PDF, TXT, DOC, DOCX, MD, PY, JS, JAVA, CPP, etc.
- Click **"Upload Files"** to process them

### 4. Configure Training

- Set **Number of Epochs** (default: 5)
- Set **Learning Rate** (default: 0.1)
- Click **"Start Training"**

Watch the real-time progress bar as the system trains!

### 5. Query the System

Once training is complete:
- Enter a query in the text box
- Examples:
  - "Write a story about artificial intelligence"
  - "Calculate 25 + 17"
  - "Create a Python function to sort a list"
- Press **Enter** or click **"Process Query"**
- View the response and routing information

### 6. Visualize Graphs

- Select a module tab (Text, Math, or Code)
- Click **"Load Graph"**
- Interact with the visualization:
  - Drag nodes to reposition
  - Click nodes for details
  - Export to various formats

---

## 📸 Dashboard Overview

### Main Sections

1. **Header**
   - System status indicator (🟢 Ready, 🟠 Training, 🔴 Not Ready)
   - Real-time status messages

2. **File Upload Panel**
   - Drag-and-drop upload zone
   - File list with sizes
   - Upload and clear controls

3. **Training Control Panel**
   - Training parameter configuration
   - Progress tracking
   - Training statistics display

4. **Query Interface**
   - Text input for queries
   - Response display area
   - Module routing information

5. **Statistics Dashboard**
   - Total nodes across all modules
   - Total edges across all modules
   - Query counter
   - Training status

6. **Graph Visualization**
   - Module selector tabs
   - Interactive D3.js graph
   - Export controls
   - Graph statistics overlay

---

## 🎯 Usage Examples

### Example 1: Training on Research Papers (PDFs)

1. Upload your PDF research papers
2. Set epochs to 10 for thorough training
3. Click "Start Training"
4. Wait for completion (progress shown in real-time)
5. Query: "Explain quantum computing concepts"

### Example 2: Training on Code Repository

1. Upload Python files from your codebase
2. Set learning rate to 0.15
3. Train the system
4. Query: "Write a function to implement binary search"

### Example 3: Mixed Training Data

1. Upload a mix of:
   - Text documents (TXT, PDF)
   - Code files (PY, JS)
   - Documentation (MD)
2. Train with default settings
3. The system automatically categorizes by domain
4. Query different modules:
   - Text queries → Text module
   - Math queries → Math module
   - Code queries → Code module

---

## 🔧 API Endpoints

The dashboard provides REST API endpoints:

### System Control
- `GET /api/status` - Get system status
- `POST /api/initialize` - Initialize the system

### File Management
- `POST /api/upload` - Upload files
- `POST /api/clear_uploads` - Clear uploaded files

### Training
- `POST /api/train` - Start training
  - Body: `{"epochs": 5, "learning_rate": 0.1}`

### Query Processing
- `POST /api/query` - Process a query
  - Body: `{"query": "your question here"}`

### Visualization
- `GET /api/graph/<module_name>` - Get graph data for visualization
- `GET /api/export/<module_name>/<format>` - Export graph (json/graphml/dot)
- `GET /api/statistics` - Get detailed statistics

---

## 🎨 Customization

### Modify Upload Limits

Edit in `dashboard.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Change to desired size
```

### Add File Types

Edit in `dashboard.py`:
```python
app.config['ALLOWED_EXTENSIONS'] = {
    'txt', 'pdf', 'doc', 'docx', 'md',  # Add your extensions here
}
```

### Change Port

Edit in `dashboard.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port number
```

---

## 💡 Tips & Best Practices

### Training Tips
1. **Start small**: Test with 10-20 files first
2. **Adjust epochs**: More epochs = better learning, but slower
3. **Learning rate**: 
   - Higher (0.15-0.3) = faster learning, may be unstable
   - Lower (0.05-0.1) = slower but more stable
4. **Mixed data**: Upload diverse content for better generalization

### Query Tips
1. **Be specific**: "Write a function to sort numbers" works better than "write code"
2. **Natural language**: Write queries as you would ask a person
3. **Check routing**: See which module handled your query and confidence score

### Visualization Tips
1. **Load graph after training**: Graphs are only available after training
2. **Limit nodes**: Large graphs (>500 nodes) may be slow
3. **Export for analysis**: Use GraphML export for Gephi/Cytoscape analysis
4. **Drag nodes**: Rearrange the graph to see structure better

---

## 🐛 Troubleshooting

### Dashboard won't start
- Check if port 5000 is available
- Ensure Flask is installed: `pip install flask`
- Check error messages in terminal

### PDF processing fails
- Install PyPDF2: `pip install PyPDF2`
- Some PDFs (encrypted, scanned images) can't be processed
- Try converting to TXT first

### Training takes too long
- Reduce number of epochs
- Upload fewer files
- Use smaller learning rate
- Check system resources (CPU, memory)

### Graph won't load
- Ensure module is trained
- Check browser console for errors
- Refresh the page
- Try with fewer nodes

### Query returns error
- Ensure system is trained first
- Check training completed successfully
- Verify query format
- Check response in browser console

---

## 📚 Related Documentation

- `README.md` - Main project overview
- `QUICKSTART.md` - Command-line usage
- `PROJECT_DOCUMENTATION.md` - Technical details
- `main.py` - Command-line demo

---

## 🔒 Security Notes

**For Development Use**: This dashboard is designed for local development and testing.

For production deployment:
- Add authentication
- Implement file upload validation
- Add rate limiting
- Use HTTPS
- Sanitize user inputs
- Implement CSRF protection

---

## 🎉 Features at a Glance

✅ Drag-and-drop file upload  
✅ PDF & DOC text extraction  
✅ Real-time training progress  
✅ Interactive query interface  
✅ Module routing display  
✅ D3.js graph visualization  
✅ Multiple export formats  
✅ Statistics dashboard  
✅ Background training  
✅ Multi-module support  
✅ Responsive design  
✅ Beautiful gradient UI  

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main project documentation
3. Check the browser console for errors
4. Verify all dependencies are installed

---

**Happy Training! 🚀**

Access your dashboard at: **http://localhost:5000**

