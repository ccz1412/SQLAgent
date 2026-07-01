# Frontend - Streamlit Web Interface

## 📋 Overview

This is the web-based user interface for the **Multi-Turn Text-to-SQL Agent** system. It provides a chat-like interface for users to interact with databases using natural language.

## ✨ Features

- **Chat Interface**: Natural conversation with the Agent
- **Real-time SQL Generation**: See generated SQL instantly
- **Result Visualization**: View query results in a formatted table
- **Model Toggle**: Switch between API mode (Zhipu AI) and local small model mode (Llama-3.1-8B)
- **Database Selection**: Choose from available Spider/BIRD databases
- **Conversation History**: Sidebar shows past queries
- **Follow-up Detection**: Automatically detects and handles follow-up questions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd E:\LLM_code_general\sqlcode-master
pip install streamlit pandas
```

### 2. Run the App

```bash
streamlit run frontend/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

### 3. Configure in Sidebar

- **Select database**: Choose from available Spider databases
- **Select model mode**:
  - `API (Zhipu AI)`: Uses Zhipu AI API (glm-4-flash)
  - `Local Small Model`: Uses your fine-tuned Llama-3.1-8B + LoRA
- **Reset conversation**: Click to start a new conversation

### 4. Start Chatting

Type questions in natural language:

```
You: List all students
Agent: [Generates SQL] SELECT * FROM students; [Executes] [Returns results]

You: Only from Computer Science department  ← Follow-up detected!
Agent: [Modifies previous SQL] ...
```

## 📂 Directory Structure

```
frontend/
├── streamlit_app.py      # Main Streamlit application
├── README.md             # This file
└── assets/              # (Optional) Images, CSS, etc.
```

## 🎨 UI Components

### Chat Messages

- **User messages**: Shown on the right, with user icon
- **Agent messages**: Shown on the left, with assistant icon
  - Includes generated SQL (in code block)
  - Includes execution results (in table)
  - Includes error messages (if any)

### Sidebar

- **Database Selection**: Dropdown with available databases
- **Model Selection**: Radio button to switch between API and local model
- **Reset Button**: Clear conversation history
- **History**: Collapsible list of past queries

## ⚙️ Configuration

### Database Paths

The app looks for databases in these locations:

1. `data/spider_databases/{db_id}/{db_id}.sqlite` (Spider dataset, read from `config/db_config.yaml`)
2. `test_databases/{db_id}.sqlite` (Custom test databases)

To add a new database:
```bash
# Copy your SQLite file to test_databases/
cp your_db.sqlite test_databases/
```

### API Configuration

Edit `config/api_config.yaml`:

```yaml
api:
  base_url: "https://open.bigmodel.cn/api/paas/v4"
  model_name: "glm-4-flash"
  api_key: "your-api-key"
```

### Local Model Configuration

Edit `config/model_config.yaml`:

```yaml
small_model:
  model_path: "dep/model/Meta-Llama-3___1-8B-Instruct"
  lora_path: "exp/outputs/sql2sr_lora"
  load_in_4bit: true
```

## 🧪 Testing

### Test 1: API Mode (Quick)

```bash
# 1. Set API key in config/api_config.yaml
# 2. Run Streamlit
streamlit run frontend/streamlit_app.py

# 3. In sidebar, select "API (Zhipu AI)" mode
# 4. Type: "List all students"
```

### Test 2: Local Model Mode (Requires GPU)

```bash
# 1. Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Install Transformers
pip install transformers peft accelerate bitsandbytes

# 3. Run Streamlit
streamlit run frontend/streamlit_app.py

# 4. In sidebar, select "Local Small Model" mode
# 5. Type: "List all students"
```

## 🐛 Troubleshooting

### Issue: "No databases found"

**Solution**: 
- Check that `data/spider_databases/` exists and contains `.sqlite` files
- Or add custom databases to `test_databases/`

### Issue: "DialogueManager initialization failed"

**Solution**:
- Check that `src/dialogue/dialogue_manager.py` exists
- Check that all dependencies are installed (`pip install -r requirements.txt`)
- Check the error message in the Streamlit UI

### Issue: "API connection failed"

**Solution**:
- Verify API key in `config/api_config.yaml`
- Check network connection
- Try a simple test: `python -c "from src.call_api.qwen_client import chat; chat([{'role':'user', 'content':'Hi'}])"`

### Issue: "CUDA out of memory"

**Solution**:
- Use 4-bit quantization (set `load_in_4bit: true` in config)
- Reduce batch size
- Use API mode instead of local model

## 📊 Example Conversation

```
┌──────────────────────────────────────────────────────┐
│  🤖 Text-to-SQL Agent                        ⚙️ │
├──────────────────────────────────────────────────────┤
│                                              [sidebar]
│  You: List all students                        >
│                                              >
│  Agent: [SQL] SELECT * FROM students;         >
│          [Results] 3 rows                     >
│                                              >
│  You: Only from CS department                 >
│                                              >
│  Agent: [Follow-up detected]                 >
│          [SQL] SELECT * FROM students         >
│                WHERE major = 'CS';             >
│          [Results] 2 rows                     >
│                                              >
│  [                         Type a message...] │
└──────────────────────────────────────────────────────┘
```

## 🔧 Development

### Modify UI Styling

Edit the `st.markdown()` CSS section in `streamlit_app.py`:

```python
st.markdown("""
<style>
    .main-header { ... }
    .sql-box { ... }
    ...
</style>
""", unsafe_allow_html=True)
```

### Add New Features

1. **SQL syntax highlighting**: Use `st.code(sql, language="sql")`
2. **Export results**: Add a "Download as CSV" button
3. **Multi-database comparison**: Allow selecting multiple databases
4. **User authentication**: Add login/session management

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Project README](../README_V2.md)

## 📝 Notes

- The frontend communicates with the backend through `src/dialogue/dialogue_manager.py`
- All configuration is read from `config/` directory
- The app runs on port 8501 by default (can be changed with `--server.port`)
