"""
长期记忆模块

功能：
1. 短期记忆（当前对话的历史）
2. 长期记忆（向量数据库 ChromaDB）
3. Schema 记忆（数据库结构缓存）
4. 纠错经验记忆（历史纠错案例）

记忆策略：
- 短期记忆：直接存储在内存中（列表）
- 长期记忆：存入 ChromaDB，支持语义检索
- Schema 记忆：缓存数据库 schema，避免重复查询
- 纠错经验：存储 (错误SQL, 错误类型, 纠正后SQL) 三元组

使用示例：
    from src.memory.memory_manager import MemoryManager
    
    mm = MemoryManager(db_id="student_db")
    
    # 存储对话历史
    mm.add_turn(question="列出所有学生", sql="SELECT * FROM students", result=...)
    
    # 检索相关历史
    relevant = mm.search_relevant("学生名单")
    
    # 存储纠错经验
    mm.add_correction_experience(
        wrong_sql="...",
        error_type="WHERE",
        corrected_sql="..."
    )
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    记忆管理器（统一接口）
    
    职责：
    1. 管理短期和长期记忆
    2. 提供统一的存储和检索接口
    3. 支持多种记忆类型
    """
    
    def __init__(self, db_id: str, use_chroma: bool = False):
        """
        初始化记忆管理器
        
        Args:
            db_id: 数据库 ID
            use_chroma: 是否使用 ChromaDB（需要安装 chromadb）
        """
        self.db_id = db_id
        self.use_chroma = use_chroma
        
        # 短期记忆（当前对话）
        self.short_term: List[Dict] = []
        
        # 长期记忆（ChromaDB）
        self.long_term = None
        if use_chroma:
            self._init_chroma()
        
        # Schema 记忆（缓存）
        self.schema_cache: Optional[str] = None
        
        # 纠错经验记忆
        self.correction_experiences: List[Dict] = []
        
        logger.info(f"记忆管理器初始化完成（db_id={db_id}, use_chroma={use_chroma}）")
    
    def _init_chroma(self):
        """初始化 ChromaDB（可选）"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # 创建 Chroma 客户端（持久化到磁盘）
            persist_directory = project_root / "data" / "chroma_db"
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_directory)
            )
            
            # 创建或获取 collection
            collection_name = f"dialogue_{self.db_id}"
            self.long_term = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": f"Dialogue history for {self.db_id}"}
            )
            
            logger.info(f"ChromaDB 初始化成功（collection: {collection_name}）")
        
        except ImportError:
            logger.warning("未安装 chromadb，长期记忆功能不可用")
            self.use_chroma = False
        except Exception as e:
            logger.error(f"ChromaDB 初始化失败: {e}")
            self.use_chroma = False
    
    # ==================== 短期记忆 ====================
    
    def add_turn(self, question: str, sql: Optional[str] = None, result: Optional[Any] = None):
        """
        添加一轮对话到短期记忆
        
        Args:
            question: 用户问题
            sql: 生成的 SQL（可选）
            result: 执行结果（可选）
        """
        turn = {
            "turn_id": len(self.short_term) + 1,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "sql": sql,
            "result": result
        }
        
        self.short_term.append(turn)
        logger.debug(f"添加到短期记忆：turn_{turn['turn_id']}")
        
        # 如果启用了 ChromaDB，也存入长期记忆
        if self.use_chroma and self.long_term is not None:
            self._store_to_chroma(turn)
    
    def get_recent_turns(self, n: int = 5) -> List[Dict]:
        """
        获取最近的 N 轮对话
        
        Args:
            n: 返回的轮数
        
        Returns:
            最近的 N 轮对话（按时间顺序）
        """
        return self.short_term[-n:]
    
    def get_all_turns(self) -> List[Dict]:
        """获取所有对话历史"""
        return self.short_term
    
    # ==================== 长期记忆（ChromaDB） ====================
    
    def _store_to_chroma(self, turn: Dict):
        """存储到 ChromaDB"""
        if self.long_term is None:
            return
        
        try:
            # 构造文档内容
            document = f"Question: {turn['question']}\n"
            if turn['sql']:
                document += f"SQL: {turn['sql']}\n"
            if turn['result']:
                document += f"Result: {turn['result']}\n"
            
            # 添加到 collection
            self.long_term.add(
                documents=[document],
                metadatas=[{
                    "turn_id": turn['turn_id'],
                    "timestamp": turn['timestamp'],
                    "db_id": self.db_id
                }],
                ids=[f"turn_{turn['turn_id']}"]
            )
            
            logger.debug(f"存储到 ChromaDB：turn_{turn['turn_id']}")
        
        except Exception as e:
            logger.error(f"存储到 ChromaDB 失败: {e}")
    
    def search_relevant(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        检索相关的历史对话
        
        Args:
            query: 查询文本
            n_results: 返回的最相关结果数
        
        Returns:
            相关的历史对话列表
        """
        if not self.use_chroma or self.long_term is None:
            logger.warning("ChromaDB 未启用，使用短期记忆进行检索")
            return self._search_short_term(query, n_results)
        
        try:
            results = self.long_term.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # 解析结果
            relevant = []
            for i, doc in enumerate(results['documents'][0]):
                relevant.append({
                    "rank": i + 1,
                    "content": doc,
                    "metadata": results['metadatas'][0][i]
                })
            
            logger.info(f"检索到 {len(relevant)} 条相关历史")
            return relevant
        
        except Exception as e:
            logger.error(f"ChromaDB 检索失败: {e}")
            return []
    
    def _search_short_term(self, query: str, n_results: int) -> List[Dict]:
        """在短期记忆中检索（简化版：关键词匹配）"""
        relevant = []
        query_lower = query.lower()
        
        for turn in reversed(self.short_term):  # 从最近开始
            question_lower = turn['question'].lower()
            
            # 简化：检查是否有共同关键词
            if any(word in question_lower for word in query_lower.split()):
                relevant.append({
                    "rank": len(relevant) + 1,
                    "content": f"Question: {turn['question']}\nSQL: {turn['sql']}",
                    "metadata": {"turn_id": turn['turn_id']}
                })
                
                if len(relevant) >= n_results:
                    break
        
        return relevant
    
    # ==================== Schema 记忆 ====================
    
    def cache_schema(self, schema: str):
        """
        缓存数据库 schema
        
        Args:
            schema: 数据库 schema 字符串
        """
        self.schema_cache = schema
        logger.info("Schema 已缓存到记忆")
    
    def get_cached_schema(self) -> Optional[str]:
        """
        获取缓存的 schema
        
        Returns:
            缓存的 schema，如果没有则返回 None
        """
        return self.schema_cache
    
    # ==================== 纠错经验记忆 ====================
    
    def add_correction_experience(self, wrong_sql: str, error_type: str, corrected_sql: str):
        """
        添加纠错经验到记忆
        
        Args:
            wrong_sql: 错误的 SQL
            error_type: 错误类型（WHERE, JOIN, etc.）
            corrected_sql: 纠正后的 SQL
        """
        experience = {
            "timestamp": datetime.now().isoformat(),
            "wrong_sql": wrong_sql,
            "error_type": error_type,
            "corrected_sql": corrected_sql
        }
        
        self.correction_experiences.append(experience)
        logger.debug(f"添加纠错经验：{error_type}")
        
        # 如果启用了 ChromaDB，也存入
        if self.use_chroma and self.long_term is not None:
            try:
                document = f"Error Type: {error_type}\nWrong SQL: {wrong_sql}\nCorrected SQL: {corrected_sql}"
                
                self.long_term.add(
                    documents=[document],
                    metadatas=[{
                        "type": "correction_experience",
                        "error_type": error_type,
                        "timestamp": experience['timestamp']
                    }],
                    ids=[f"correction_{len(self.correction_experiences)}"]
                )
            except Exception as e:
                logger.error(f"存储纠错经验到 ChromaDB 失败: {e}")
    
    def search_similar_corrections(self, wrong_sql: str, n_results: int = 3) -> List[Dict]:
        """
        检索相似的纠错案例
        
        Args:
            wrong_sql: 错误的 SQL
            n_results: 返回的最相似结果数
        
        Returns:
            相似的纠错案例列表
        """
        if not self.use_chroma:
            # 简化：返回最近的纠错经验
            return self.correction_experiences[-n_results:]
        
        try:
            results = self.long_term.query(
                query_texts=[wrong_sql],
                n_results=n_results,
                where={"type": "correction_experience"}  # 只检索纠错经验
            )
            
            similar = []
            for i, doc in enumerate(results['documents'][0]):
                similar.append({
                    "rank": i + 1,
                    "content": doc,
                    "metadata": results['metadatas'][0][i]
                })
            
            logger.info(f"检索到 {len(similar)} 条相似纠错案例")
            return similar
        
        except Exception as e:
            logger.error(f"检索相似纠错案例失败: {e}")
            return []
    
    # ==================== 持久化 ====================
    
    def save(self, path: Optional[str] = None):
        """
        保存记忆到磁盘
        
        Args:
            path: 保存路径（默认：data/memory/{db_id}.json）
        """
        if path is None:
            path = project_root / "data" / "memory" / f"{self.db_id}.json"
        
        import json
        
        data = {
            "db_id": self.db_id,
            "short_term": self.short_term,
            "correction_experiences": self.correction_experiences,
            "schema_cache": self.schema_cache
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"记忆已保存 to {path}")
    
    def load(self, path: Optional[str] = None):
        """
        从磁盘加载记忆
        
        Args:
            path: 加载路径（默认：data/memory/{db_id}.json）
        """
        if path is None:
            path = project_root / "data" / "memory" / f"{self.db_id}.json"
        
        if not Path(path).exists():
            logger.warning(f"记忆文件不存在：{path}")
            return
        
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.short_term = data.get("short_term", [])
        self.correction_experiences = data.get("correction_experiences", [])
        self.schema_cache = data.get("schema_cache")
        
        logger.info(f"记忆已加载 from {path}（{len(self.short_term)} turns）")


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("记忆管理器测试")
    print("=" * 60)
    
    mm = MemoryManager(db_id="test_db", use_chroma=False)
    
    # 测试 1：添加对话
    print("\n[测试 1] 添加对话到短期记忆")
    mm.add_turn(question="列出所有学生", sql="SELECT * FROM students", result=[...])
    mm.add_turn(question="只看计算机系的", sql="SELECT * FROM students WHERE major='CS'", result=[...])
    print(f"  短期记忆中有 {len(mm.get_all_turns())} 轮对话")
    
    # 测试 2：检索相关历史
    print("\n[测试 2] 检索相关历史")
    relevant = mm.search_relevant("学生名单", n_results=2)
    print(f"  检索到 {len(relevant)} 条相关历史")
    for r in relevant:
        print(f"    Rank {r['rank']}: {r['content'][:50]}...")
    
    # 测试 3：缓存 Schema
    print("\n[测试 3] 缓存 Schema")
    mm.cache_schema("Table: students (id, name, major)")
    schema = mm.get_cached_schema()
    print(f"  缓存的 Schema: {schema[:50]}...")
    
    # 测试 4：添加纠错经验
    print("\n[测试 4] 添加纠错经验")
    mm.add_correction_experience(
        wrong_sql="SELECT * FROM students WHERE",
        error_type="WHERE",
        corrected_sql="SELECT * FROM students WHERE 1=1"
    )
    print(f"  纠错经验数：{len(mm.correction_experiences)}")
    
    # 测试 5：保存和加载
    print("\n[测试 5] 保存和加载记忆")
    mm.save()
    
    mm2 = MemoryManager(db_id="test_db", use_chroma=False)
    mm2.load()
    print(f"  加载后短期记忆轮数：{len(mm2.get_all_turns())}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
