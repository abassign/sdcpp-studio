"""
Workers View Manager - Query execution and view management
Handles semantic queries against the tags database and manages saved views.
"""
import os
import sys
import json
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime

sys.path.insert(0, os.getcwd())

# Configuration
VIEWS_DB_PATH = "/app/data/saved_views.json"
os.makedirs(os.path.dirname(VIEWS_DB_PATH), exist_ok=True)


class QueryExecutor:
    """Execute semantic queries against tags DB"""
    
    # Supported operators
    OPERATORS = {
        "equals": lambda tag_val, query_val: str(tag_val).lower() == str(query_val).lower(),
        "contains": lambda tag_val, query_val: str(query_val).lower() in str(tag_val).lower(),
        "starts_with": lambda tag_val, query_val: str(tag_val).lower().startswith(str(query_val).lower()),
        "ends_with": lambda tag_val, query_val: str(tag_val).lower().endswith(str(query_val).lower()),
        "in": lambda tag_val, query_val: str(tag_val).lower() in [v.lower() for v in query_val] if isinstance(query_val, list) else False,
        "not_equals": lambda tag_val, query_val: str(tag_val).lower() != str(query_val).lower(),
        "not_contains": lambda tag_val, query_val: str(query_val).lower() not in str(tag_val).lower(),
        "gt": lambda tag_val, query_val: float(tag_val) > float(query_val) if _is_number(tag_val) and _is_number(query_val) else False,
        "lt": lambda tag_val, query_val: float(tag_val) < float(query_val) if _is_number(tag_val) and _is_number(query_val) else False,
    }
    
    def __init__(self, tags_db: Dict):
        self.tags_db = tags_db
    
    def execute_query(self, query: Dict) -> List[str]:
        """
        Execute a query and return matching image keys.
        
        Query format:
        {
            "logic": "AND|OR",
            "conditions": [
                {"field": "tags", "category": "subject", "operator": "contains", "value": "landscape"},
                {"field": "summary", "operator": "contains", "value": "mountain"},
                {"field": "manual_tags", "operator": "contains", "value": "favorite"}
            ]
        }
        """
        results = []
        logic = query.get("logic", "OR").upper()
        conditions = query.get("conditions", [])
        
        if not conditions:
            return list(self.tags_db.get("tags", {}).keys())
        
        for img_key, img_data in self.tags_db.get("tags", {}).items():
            matches = []
            
            for condition in conditions:
                match = self._evaluate_condition(img_data, condition)
                matches.append(match)
            
            # Apply logic
            if logic == "AND":
                if all(matches):
                    results.append(img_key)
            elif logic == "OR":
                if any(matches):
                    results.append(img_key)
            elif logic == "NOT":
                if not any(matches):
                    results.append(img_key)
        
        return results
    
    def _evaluate_condition(self, img_data: Dict, condition: Dict) -> bool:
        """Evaluate single condition against image data"""
        field = condition.get("field", "summary")
        operator = condition.get("operator", "contains")
        value = condition.get("value", "")
        category = condition.get("category")  # For tags field
        
        op_func = self.OPERATORS.get(operator)
        if not op_func:
            return False
        
        # Handle different fields
        if field == "manual_tags":
            tags_list = img_data.get("manual_tags", [])
            for tag in tags_list:
                if op_func(tag, value):
                    return True
            return False
        
        elif field == "summary":
            summary = img_data.get("summary", "")
            return op_func(summary, value)
        
        elif field == "tags":
            # Search within tags by category
            for tag in img_data.get("tags", []):
                if category and tag.get("category") != category:
                    continue
                tag_value = tag.get("value", "")
                if op_func(tag_value, value):
                    return True
            return False
        
        elif field == "generated_by":
            return op_func(img_data.get("generated_by", ""), value)
        
        elif field == "timestamp":
            return op_func(img_data.get("timestamp", ""), value)
        
        elif field == "confidence":
            # Check if any tag has confidence matching
            for tag in img_data.get("tags", []):
                conf = tag.get("confidence", 0)
                if op_func(conf, value):
                    return True
            return False
        
        return False
    
    def natural_language_to_query(self, nl_query: str) -> Dict:
        """
        Convert natural language query to structured query.
        
        Examples:
        - "sport athlete" -> OR conditions for sport OR athlete
        - "landscape mountain" -> OR conditions
        - "not anime" -> NOT condition for anime
        """
        keywords = nl_query.lower().split()
        
        # Handle "not" prefix
        not_keywords = []
        positive_keywords = []
        
        i = 0
        while i < len(keywords):
            if keywords[i] == "not" and i + 1 < len(keywords):
                not_keywords.append(keywords[i + 1])
                i += 2
            else:
                positive_keywords.append(keywords[i])
                i += 1
        
        conditions = []
        
        # Add positive conditions
        for kw in positive_keywords:
            conditions.append({
                "field": "summary",
                "operator": "contains",
                "value": kw
            })
            conditions.append({
                "field": "tags",
                "operator": "contains",
                "value": kw
            })
        
        # Add negative conditions
        for kw in not_keywords:
            conditions.append({
                "field": "summary",
                "operator": "not_contains",
                "value": kw
            })
        
        # Determine logic
        if not_keywords and not positive_keywords:
            logic = "NOT"
        elif len(conditions) > 1:
            logic = "OR"
        else:
            logic = "OR"
        
        return {
            "logic": logic,
            "conditions": conditions
        }

    def natural_language_to_query_llm(self, nl_query: str, model: str) -> Dict:
        """Use LLM to translate natural language search into structured query."""
        import subprocess
        import re
        from workers.i2t_llama import find_gguf_model
        
        prompt = f"""Convert the following search query into a JSON search condition. 
User query: '{nl_query}'

The JSON object must follow this exact schema:
{{
    "logic": "OR",
    "conditions": [
        {{"field": "summary", "operator": "contains", "value": "word"}}
    ]
}}
Extract the core subject/keywords. Ignore conversational words like "cerca", "i", "prompt", "con", "trova". For example, if query is "cerca i prompt con boyscout", the value is just "boyscout".
If using "non", use operator "not_contains" and logic "NOT" or "AND".
Return ONLY valid JSON."""

        gguf_model, _ = find_gguf_model()
        if not gguf_model:
            raise ValueError("No GGUF model found for LLM search.")

        cmd = [
            "llama-cli",
            "-m", gguf_model,
            "-p", prompt,
            "--temp", "0.1",
            "-n", "200",
            "-ngl", "99",
            "--no-display-prompt",
            "--simple-io"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env={**os.environ, "HIP_VISIBLE_DEVICES": "0"})
        if result.returncode != 0:
            raise ValueError(f"LLM failed: {result.stderr}")
            
        output = result.stdout.strip()
        # Extract JSON
        json_match = re.search(r'(\{[\s\S]*\})', output)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "logic" in parsed and "conditions" in parsed:
                    print(f"--> [ViewManager] LLM parsed query successfully: {parsed}", flush=True)
                    return parsed
            except Exception:
                pass
        
        raise ValueError(f"Could not extract valid JSON from LLM output: {output}")


def _is_number(val) -> bool:
    """Check if value is a number"""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


class ViewManager:
    """Manage saved views and queries"""
    
    def __init__(self):
        self.views_db = self._load_views_db()
    
    def _load_views_db(self) -> Dict:
        """Load views database"""
        if os.path.exists(VIEWS_DB_PATH):
            try:
                with open(VIEWS_DB_PATH, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": "1.0", "views": []}
    
    def _save_views_db(self):
        """Save views database"""
        with open(VIEWS_DB_PATH, 'w') as f:
            json.dump(self.views_db, f, indent=2)
    
    def create_view(self, name: str, description: str, query: Dict, 
                   color: str = "#2196F3") -> str:
        """Create a new saved view"""
        view_id = f"view_{len(self.views_db['views']) + 1:03d}"
        
        view = {
            "id": view_id,
            "name": name,
            "description": description,
            "query": query,
            "created": self._timestamp(),
            "last_used": self._timestamp(),
            "color": color
        }
        
        self.views_db["views"].append(view)
        self._save_views_db()
        
        return view_id
    
    def update_view(self, view_id: str, updates: Dict) -> bool:
        """Update an existing view"""
        for view in self.views_db["views"]:
            if view["id"] == view_id:
                view.update({k: v for k, v in updates.items() if v is not None})
                view["last_used"] = self._timestamp()
                self._save_views_db()
                return True
        return False
    
    def delete_view(self, view_id: str) -> bool:
        """Delete a view"""
        original_len = len(self.views_db["views"])
        self.views_db["views"] = [v for v in self.views_db["views"] if v["id"] != view_id]
        
        if len(self.views_db["views"]) < original_len:
            self._save_views_db()
            return True
        return False
    
    def get_view(self, view_id: str) -> Optional[Dict]:
        """Get a view by ID"""
        for view in self.views_db["views"]:
            if view["id"] == view_id:
                return view
        return None
    
    def execute_view(self, view_id: str, tags_db: Dict) -> List[str]:
        """Execute a saved view and return matching image keys"""
        view = self.get_view(view_id)
        if not view:
            return []
        
        executor = QueryExecutor(tags_db)
        results = executor.execute_query(view["query"])
        
        # Update last_used
        view["last_used"] = self._timestamp()
        self._save_views_db()
        
        return results
    
    def execute_query(self, query: Dict, tags_db: Dict) -> List[str]:
        """Execute a query directly (without saving as view)"""
        executor = QueryExecutor(tags_db)
        return executor.execute_query(query)
    
    def natural_language_query(self, nl_query: str, tags_db: Dict, model: Optional[str] = None) -> Tuple[Dict, List[str]]:
        """
        Execute a natural language query.
        Returns (query, results)
        """
        executor = QueryExecutor(tags_db)
        
        query = None
        if model:
            try:
                print(f"--> [ViewManager] Attempting LLM parsing with {model}...", flush=True)
                query = executor.natural_language_to_query_llm(nl_query, model)
            except Exception as e:
                print(f"--> [ViewManager] LLM query parsing failed: {e}. Falling back to simple query.", flush=True)
                query = None
                
        if not query:
            query = executor.natural_language_to_query(nl_query)
            
        results = executor.execute_query(query)
        return query, results
    
    def list_views(self) -> List[Dict]:
        """List all saved views"""
        return self.views_db.get("views", [])
    
    @staticmethod
    def _timestamp():
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    # Test mode
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage semantic views")
    parser.add_argument("--list", action="store_true", help="List views")
    parser.add_argument("--create", nargs=2, metavar=("NAME", "QUERY"), help="Create view")
    parser.add_argument("--delete", metavar="VIEW_ID", help="Delete view")
    parser.add_argument("--test-query", metavar="QUERY", help="Test NL query")
    
    args = parser.parse_args()
    
    vm = ViewManager()
    
    if args.list:
        views = vm.list_views()
        if views:
            for v in views:
                print(f"{v['id']}: {v['name']} - {v['description']}")
        else:
            print("No views saved")
    
    elif args.create:
        name, query_str = args.create
        # Simple query construction
        query = {
            "logic": "OR",
            "conditions": [
                {"field": "summary", "operator": "contains", "value": w}
                for w in query_str.split()
            ]
        }
        view_id = vm.create_view(name, query_str, query)
        print(f"Created view: {view_id}")
    
    elif args.delete:
        if vm.delete_view(args.delete):
            print(f"Deleted view: {args.delete}")
        else:
            print(f"View not found: {args.delete}")
    
    elif args.test_query:
        from workers.tag_worker import load_tags_db
        tags_db = load_tags_db()
        query, results = vm.natural_language_query(args.test_query, tags_db)
        print(f"Query: {json.dumps(query, indent=2)}")
        print(f"Results: {len(results)} images")
        for r in results[:5]:
            print(f"  - {r}")
    
    else:
        parser.print_help()
