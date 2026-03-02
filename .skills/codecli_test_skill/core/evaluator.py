#!/usr/bin/env python3
"""
Code Agent Self-Validation System (Enhanced)
=============================================
更完善的七维度评估实现
"""

import os
import sys
import ast
import re
import json
import time
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR = Path(__file__).parent
TASKS_FILE = PROJECT_DIR / "tasks.md"
SOLUTIONS_DIR = PROJECT_DIR / "solutions"
TEST_CASES_DIR = PROJECT_DIR / "test_cases"
REPORTS_DIR = PROJECT_DIR / "reports"


# ============================================================================
# Task Parser
# ============================================================================

class TaskParser:
    """从 tasks.md 解析任务定义"""

    @staticmethod
    def parse() -> List[Dict[str, Any]]:
        """解析所有任务"""
        content = TASKS_FILE.read_text(encoding='utf-8')
        tasks = []

        pattern = r'## Task (\d+): ([^\n]+)'
        matches = list(re.finditer(pattern, content))

        for i, match in enumerate(matches):
            task_num = match.group(1)
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section = content[start:end]

            task = {
                "id": f"task_{task_num}",
                "number": int(task_num),
                "title": title,
                "requirements": TaskParser._extract_section(section, "需求"),
                "acceptance_criteria": TaskParser._extract_checklist(section),
                "test_file": TaskParser._extract_test_file(section)
            }

            # Extract difficulty
            if "⭐⭐⭐⭐" in title or "专家" in title:
                task["difficulty"] = "expert"
            elif "⭐⭐⭐" in title or "高级" in title:
                task["difficulty"] = "advanced"
            elif "⭐⭐" in title or "中级" in title:
                task["difficulty"] = "intermediate"
            else:
                task["difficulty"] = "basic"

            tasks.append(task)

        return tasks

    @staticmethod
    def _extract_section(section: str, header: str) -> List[str]:
        """提取章节内容"""
        pattern = rf'### {header}\s*(.*?)(?=###|---)'
        match = re.search(pattern, section, re.DOTALL)
        if match:
            lines = match.group(1).strip().split('\n')
            return [line.strip().lstrip('123456789. ')
                    for line in lines
                    if line.strip() and not line.startswith('###')]
        return []

    @staticmethod
    def _extract_checklist(section: str) -> List[str]:
        """提取验收标准"""
        match = re.search(r'### 验收标准\s*(.*?)(?=###|---)', section, re.DOTALL)
        if match:
            lines = match.group(1).strip().split('\n')
            criteria = []
            for line in lines:
                line = line.strip()
                if line.startswith('- ['):
                    cleaned = re.sub(r'^- \[[ x]\]\s*', '', line)
                    criteria.append(cleaned)
            return criteria
        return []

    @staticmethod
    def _extract_test_file(section: str) -> Optional[str]:
        """提取测试文件名"""
        match = re.search(r'### 测试文件\s*`([^`]+)`', section)
        return match.group(1) if match else None


# ============================================================================
# Solution Checker & Analyzer
# ============================================================================

class SolutionChecker:
    """检查并分析解决方案"""

    @staticmethod
    def find_solution_file(task: Dict) -> Optional[Path]:
        """查找解决方案文件"""
        test_file = task.get("test_file", "")
        module_name = test_file.replace("test_", "").replace(".py", "")

        possible_files = [
            SOLUTIONS_DIR / f"{module_name}.py",
            SOLUTIONS_DIR / "multi_head_attention.py",
            SOLUTIONS_DIR / "rmsnorm.py",
            SOLUTIONS_DIR / "moe_router.py",
            SOLUTIONS_DIR / f"{task['id']}.py",
        ]

        for sf in possible_files:
            if sf.exists():
                return sf
        return None

    @staticmethod
    def check_syntax(file_path: Path) -> bool:
        """检查语法是否正确"""
        try:
            ast.parse(file_path.read_text(encoding='utf-8'))
            return True
        except SyntaxError:
            return False

    @staticmethod
    def analyze_code(file_path: Path) -> Dict[str, Any]:
        """深度分析代码"""
        code = file_path.read_text(encoding='utf-8')
        tree = ast.parse(code)

        metrics = {
            "lines": len(code.split('\n')),
            "functions": [],
            "classes": [],
            "imports": [],
            "has_docstrings": False,
            "has_type_hints": False,
            "has_error_handling": False,
            "has_classes": False,
            "docstring_coverage": 0.0,
            "avg_function_length": 0.0,
            "cyclomatic_complexity": 0,
        }

        # Count functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics["functions"].append(node.name)
                if ast.get_docstring(node):
                    metrics["has_docstrings"] = True
                # Calculate function length
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    length = node.end_lineno - node.lineno
                    if hasattr(node, 'body') and node.body:
                        metrics["avg_function_length"] += length
            elif isinstance(node, ast.AsyncFunctionDef):
                metrics["functions"].append(node.name)
                if ast.get_docstring(node):
                    metrics["has_docstrings"] = True
            elif isinstance(node, ast.ClassDef):
                metrics["classes"].append(node.name)
                metrics["has_classes"] = True
                if ast.get_docstring(node):
                    metrics["has_docstrings"] = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    metrics["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    metrics["imports"].append(node.module)
            elif isinstance(node, ast.Try):
                metrics["has_error_handling"] = True
            elif isinstance(node, ast.arguments):
                if node.returns or any(arg.annotation for arg in node.args if hasattr(arg, 'annotation')):
                    metrics["has_type_hints"] = True

        # Calculate averages
        total_funcs = len(metrics["functions"]) + len(metrics["classes"])
        if total_funcs > 0:
            metrics["docstring_coverage"] = 1.0 if metrics["has_docstrings"] else 0.0
            metrics["avg_function_length"] = metrics["avg_function_length"] / total_funcs

        return metrics


# ============================================================================
# Enhanced Validators for 7 Dimensions
# ============================================================================

class DimensionValidator:
    """七维度验证器"""

    # ========================================================================
    # 维度1: 代码质量
    # ========================================================================

    @staticmethod
    def validate_code_quality(tasks: List[Dict]) -> Dict[str, Any]:
        """验证代码质量：结构、风格、文档、类型提示"""
        scores = []

        for task in tasks:
            solution_file = SolutionChecker.find_solution_file(task)
            if not solution_file:
                continue

            # Check syntax
            if not SolutionChecker.check_syntax(solution_file):
                scores.append(0.0)
                continue

            metrics = SolutionChecker.analyze_code(solution_file)
            score = 0.0

            # Documentation (30%)
            if metrics["has_docstrings"]:
                score += 0.3

            # Type hints (20%)
            if metrics["has_type_hints"]:
                score += 0.2

            # Structure (25%)
            if metrics["functions"] or metrics["classes"]:
                score += 0.25

            # Imports (15%)
            if metrics["imports"]:
                score += 0.15

            # Error handling (10%)
            if metrics["has_error_handling"]:
                score += 0.1

            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "score": avg_score,
            "details": f"Based on {len(scores)} solutions analyzed",
            "analyzed_count": len(scores),
            "level": DimensionValidator._get_level(avg_score)
        }

    # ========================================================================
    # 维度2: 意图符合度
    # ========================================================================

    @staticmethod
    def validate_intent_alignment(tasks: List[Dict]) -> Dict[str, Any]:
        """验证意图符合度：需求匹配、功能完整性"""
        total_requirements = 0
        met_requirements = 0

        for task in tasks:
            requirements = task.get("requirements", [])
            solution_file = SolutionChecker.find_solution_file(task)

            if not solution_file:
                continue

            code = solution_file.read_text(encoding='utf-8').lower()

            for req in requirements:
                total_requirements += 1
                req_lower = req.lower()
                # Extract meaningful keywords (length > 3)
                keywords = [w for w in req_lower.split() if len(w) > 3]
                if keywords:
                    matches = sum(1 for kw in keywords if kw in code)
                    if matches >= len(keywords) * 0.5:  # At least 50% keywords match
                        met_requirements += 1

        score = met_requirements / total_requirements if total_requirements > 0 else 0.0

        return {
            "score": score,
            "details": f"Requirements met: {met_requirements}/{total_requirements}",
            "met": met_requirements,
            "total": total_requirements,
            "level": DimensionValidator._get_level(score)
        }

    # ========================================================================
    # 维度3: 结果精确度
    # ========================================================================

    @staticmethod
    def validate_result_accuracy(tasks: List[Dict]) -> Dict[str, Any]:
        """验证结果精确度：输出正确性、测试通过率"""
        test_results = {}

        for task in tasks:
            test_file = task.get("test_file")
            if not test_file:
                continue

            test_path = TEST_CASES_DIR / test_file
            if not test_path.exists():
                continue

            # Run test
            try:
                result = subprocess.run(
                    [sys.executable, str(test_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(PROJECT_DIR)
                )

                test_results[task["id"]] = {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            except Exception as e:
                test_results[task["id"]] = {"success": False, "error": str(e)}

        passed = sum(1 for tr in test_results.values() if tr.get("success"))
        total = len(test_results)

        score = passed / total if total > 0 else 0.0

        return {
            "score": score,
            "details": f"Tests passed: {passed}/{total}",
            "passed": passed,
            "total": total,
            "test_results": test_results,
            "level": DimensionValidator._get_level(score)
        }

    # ========================================================================
    # 维度4: 工程能力
    # ========================================================================

    @staticmethod
    def validate_engineering_capability(tasks: List[Dict]) -> Dict[str, Any]:
        """验证工程能力：错误处理、模块化、库使用"""
        capabilities = {
            "error_handling": 0,
            "modular_design": 0,
            "external_libraries": 0,
            "testing": 0,
            "documentation": 0
        }

        total_solutions = 0

        for task in tasks:
            solution_file = SolutionChecker.find_solution_file(task)
            if not solution_file:
                continue

            total_solutions += 1
            metrics = SolutionChecker.analyze_code(solution_file)

            if metrics.get("has_error_handling"):
                capabilities["error_handling"] += 1
            if metrics.get("functions") or metrics.get("classes"):
                capabilities["modular_design"] += 1
            if metrics.get("imports"):
                capabilities["external_libraries"] += 1
            if metrics.get("has_docstrings"):
                capabilities["documentation"] += 1

        # Testing capability exists if test files exist
        capabilities["testing"] = len([t for t in tasks if t.get("test_file")])

        # Score: each capability is worth up to 0.2
        max_per_cap = total_solutions if total_solutions > 0 else 1
        score = sum(min(1.0, v / max_per_cap) for v in capabilities.values() if v > 0) / 5.0

        return {
            "score": score,
            "details": f"Capabilities: {[k for k, v in capabilities.items() if v > 0]}",
            "capabilities": capabilities,
            "level": DimensionValidator._get_level(score)
        }

    # ========================================================================
    # 维度5: 执行效率
    # ========================================================================

    @staticmethod
    def validate_execution_efficiency(tasks: List[Dict]) -> Dict[str, Any]:
        """验证执行效率：时间遵守、资源使用"""
        execution_times = []

        for task in tasks:
            test_file = task.get("test_file")
            if not test_file:
                continue

            test_path = TEST_CASES_DIR / test_file
            if not test_path.exists():
                continue

            # Measure execution time
            start = time.time()
            try:
                result = subprocess.run(
                    [sys.executable, str(test_path)],
                    capture_output=True,
                    timeout=60,
                    cwd=str(PROJECT_DIR)
                )
                elapsed = time.time() - start

                if result.returncode == 0:
                    execution_times.append(elapsed)
            except Exception:
                pass

        if not execution_times:
            return {
                "score": 0.0,
                "details": "No successful executions",
                "level": "需改进"
            }

        avg_time = sum(execution_times) / len(execution_times)
        # Assume reasonable time limit is 10 seconds per test
        efficiency = max(0.0, min(1.0, 1.0 - avg_time / 10.0))

        return {
            "score": efficiency,
            "details": f"Avg execution time: {avg_time:.2f}s, Efficiency: {efficiency:.2%}",
            "avg_time": avg_time,
            "level": DimensionValidator._get_level(efficiency)
        }

    # ========================================================================
    # 维度6: Skills 能力
    # ========================================================================

    @staticmethod
    def validate_skills_capability(tasks: List[Dict]) -> Dict[str, Any]:
        """验证 Skills 能力：工具调用、外部库使用"""
        skills_found = set()

        # Check what libraries and frameworks are used
        for task in tasks:
            solution_file = SolutionChecker.find_solution_file(task)
            if not solution_file:
                continue

            code = solution_file.read_text(encoding='utf-8').lower()

            # Detect specific skills
            if "torch" in code:
                skills_found.add("pytorch")
            if "triton" in code:
                skills_found.add("gpu_programming")
            if "cuda" in code:
                skills_found.add("cuda")
            if "numpy" in code:
                skills_found.add("numerical_computing")
            if "subprocess" in code or "multiprocessing" in code:
                skills_found.add("system_integration")

        # Skill variety bonus
        skill_count = len(skills_found)
        score = min(1.0, skill_count * 0.25)

        return {
            "score": score,
            "details": f"Skills: {list(skills_found)}" if skills_found else "No skills detected",
            "skills": list(skills_found),
            "skill_count": skill_count,
            "level": DimensionValidator._get_level(score)
        }

    # ========================================================================
    # 维度7: Multi-Agent 能力
    # ========================================================================

    @staticmethod
    def validate_multi_agent_capability(tasks: List[Dict]) -> Dict[str, Any]:
        """验证 Multi-Agent 能力：任务分解、协调"""
        # Analyze task complexity and how many are handled
        expert_count = sum(1 for t in tasks if t.get("difficulty") == "expert")
        advanced_count = sum(1 for t in tasks if t.get("difficulty") == "advanced")

        # Check if solutions exist for complex tasks
        expert_solved = 0
        advanced_solved = 0

        for task in tasks:
            if task.get("difficulty") == "expert":
                if SolutionChecker.find_solution_file(task):
                    expert_solved += 1
            elif task.get("difficulty") == "advanced":
                if SolutionChecker.find_solution_file(task):
                    advanced_solved += 1

        # Calculate coordination score
        total_complex = expert_count + advanced_count
        solved_complex = expert_solved + advanced_solved

        if total_complex > 0:
            coordination_score = solved_complex / total_complex
        else:
            coordination_score = 0.0

        # Overall score combines ability to handle and solve complex tasks
        score = (total_complex * 0.1 + solved_complex * 0.3) / max(1, total_complex)
        score = min(1.0, score)

        return {
            "score": score,
            "details": f"Expert: {expert_solved}/{expert_count}, Advanced: {advanced_solved}/{advanced_count}",
            "expert_solved": expert_solved,
            "expert_total": expert_count,
            "advanced_solved": advanced_solved,
            "advanced_total": advanced_count,
            "coordination_score": coordination_score,
            "level": DimensionValidator._get_level(score)
        }

    @staticmethod
    def _get_level(score: float) -> str:
        """获取等级描述"""
        if score >= 0.9:
            return "卓越"
        elif score >= 0.75:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        else:
            return "需改进"


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """生成 Code Agent 能力报告"""

    @staticmethod
    def generate(evaluations: Dict[str, Any]) -> str:
        """生成完整的 Markdown 报告"""

        lines = [
            "# Code Agent 能力报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 📊 总体评估",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| **总体得分** | {evaluations['overall_score']:.2f}/1.00 |",
            "",
            ReportGenerator._get_ability_level(evaluations['overall_score']),
            "",
            "---",
            "",
            "## 🎯 七维能力分析",
            ""
        ]

        dimension_names = {
            "code_quality": "1️⃣ 代码质量",
            "intent_alignment": "2️⃣ 意图符合度",
            "result_accuracy": "3️⃣ 结果精确度",
            "engineering_capability": "4️⃣ 工程能力",
            "execution_efficiency": "5️⃣ 执行效率",
            "skills_capability": "6️⃣ Skills 能力",
            "multi_agent_capability": "7️⃣ Multi-Agent 能力"
        }

        for key, name in dimension_names.items():
            dim = evaluations["dimensions"][key]
            score = dim["score"]
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))

            lines.extend([
                f"### {name}",
                "",
                f"**得分**: {score:.2f}/1.00",
                "",
                f"{bar}",
                "",
                f"**等级**: {dim['level']}",
                f"**详情**: {dim['details']}",
                ""
            ])

        # Task summary
        lines.extend([
            "---",
            "",
            "## 📋 任务完成情况",
            "",
            "| 任务 | 难度 | 实现 | 测试通过 |",
            "|------|------|------|----------|"
        ])

        for summary in evaluations["task_summary"]:
            difficulty_icon = {
                "expert": "⭐⭐⭐⭐",
                "advanced": "⭐⭐⭐",
                "intermediate": "⭐⭐",
                "basic": "⭐"
            }.get(summary.get("difficulty", ""), "⭐")

            impl = "✅" if summary.get("has_solution") else "❌"
            test = "✅" if summary.get("test_passed") else "❌"

            lines.append(
                f"| {summary.get('title', 'Unknown')} | {difficulty_icon} | {impl} | {test} |"
            )

        # Recommendations
        lines.extend([
            "",
            "---",
            "",
            "## 💡 改进建议",
            "",
            ReportGenerator._generate_recommendations(evaluations),
            "",
            "---",
            "",
            f"*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])

        return "\n".join(lines)

    @staticmethod
    def _get_ability_level(score: float) -> str:
        """获取能力等级描述"""
        if score >= 0.9:
            return "### 🏆 卓越 (Exceptional)\n\n该 Code Agent 在所有维度都表现出色，可以胜任复杂的 AI 开发任务。"
        elif score >= 0.75:
            return "### ⭐ 优秀 (Excellent)\n\n该 Code Agent 在大多数维度表现良好，能够独立完成中等至高等难度的任务。"
        elif score >= 0.6:
            return "### 👍 良好 (Good)\n\n该 Code Agent 能够完成大部分任务，但在某些复杂场景下可能需要辅助。"
        else:
            return "### ⚠️ 需改进 (Needs Improvement)\n\n该 Code Agent 在多个维度存在不足，需要进一步优化和训练。"

    @staticmethod
    def _generate_recommendations(evaluations: Dict[str, Any]) -> str:
        """生成改进建议"""
        recommendations = []

        for key, dim in evaluations["dimensions"].items():
            if dim["score"] < 0.6:
                if "code_quality" in key:
                    recommendations.append("- **代码质量**: 加强文档编写，添加类型提示，改进代码结构")
                elif "intent" in key:
                    recommendations.append(f"- **需求理解**: 提高需求满足率 (当前 {dim['score']:.2%})")
                elif "accuracy" in key:
                    recommendations.append(f"- **结果精确度**: 提高测试通过率 (当前 {dim['details']})")
                elif "engineering" in key:
                    recommendations.append("- **工程能力**: 加强错误处理和模块化设计")
                elif "efficiency" in key:
                    recommendations.append("- **执行效率**: 优化代码性能，减少执行时间")
                elif "skills" in key:
                    recommendations.append("- **工具使用**: 扩展可用工具和库的使用")
                elif "multi_agent" in key:
                    recommendations.append("- **协调能力**: 提高复杂任务的处理能力")

        if not recommendations:
            return "✅ 在所有维度表现良好，继续保持！"

        return "\n".join(recommendations)


# ============================================================================
# Main System
# ============================================================================

def main():
    """主执行流程"""

    print("=" * 70)
    print(" " * 18 + "Code Agent Self-Validation")
    print("=" * 70)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create directories
    SOLUTIONS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # Step 1: Parse tasks
    print(f"\n[1/6] 解析任务定义...")
    tasks = TaskParser.parse()
    print(f"      解析到 {len(tasks)} 个任务")

    # Step 2: Check solutions
    print(f"\n[2/6] 检查解决方案...")
    solution_status = {}
    for task in tasks:
        has_solution = SolutionChecker.find_solution_file(task) is not None
        solution_status[task["id"]] = has_solution
        status = "✅" if has_solution else "❌"
        print(f"      {status} Task {task['number']}: {task['title']}")

    # Step 3: Validate dimensions
    print(f"\n[3/6] 七维度评估...")

    validator = DimensionValidator()
    dimensions = {}

    dimensions["code_quality"] = validator.validate_code_quality(tasks)
    print(f"      1️⃣  代码质量: {dimensions['code_quality']['score']:.2f}")

    dimensions["intent_alignment"] = validator.validate_intent_alignment(tasks)
    print(f"      2️⃣  意图符合度: {dimensions['intent_alignment']['score']:.2f}")

    dimensions["result_accuracy"] = validator.validate_result_accuracy(tasks)
    print(f"      3️⃣  结果精确度: {dimensions['result_accuracy']['score']:.2f}")

    dimensions["engineering_capability"] = validator.validate_engineering_capability(tasks)
    print(f"      4️⃣  工程能力: {dimensions['engineering_capability']['score']:.2f}")

    dimensions["execution_efficiency"] = validator.validate_execution_efficiency(tasks)
    print(f"      5️⃣  执行效率: {dimensions['execution_efficiency']['score']:.2f}")

    dimensions["skills_capability"] = validator.validate_skills_capability(tasks)
    print(f"      6️⃣  Skills 能力: {dimensions['skills_capability']['score']:.2f}")

    dimensions["multi_agent_capability"] = validator.validate_multi_agent_capability(tasks)
    print(f"      7️⃣  Multi-Agent 能力: {dimensions['multi_agent_capability']['score']:.2f}")

    # Step 4: Build task summary
    print(f"\n[4/6] 构建任务摘要...")
    task_summary = []
    for task in tasks:
        task_id = task["id"]
        test_result = dimensions["result_accuracy"].get("test_results", {}).get(task_id, {})

        summary = {
            "id": task["id"],
            "title": task["title"],
            "difficulty": task.get("difficulty", "unknown"),
            "has_solution": solution_status.get(task_id, False),
            "test_passed": test_result.get("success", False)
        }
        task_summary.append(summary)

    # Step 5: Calculate overall score
    print(f"\n[5/6] 计算总体得分...")
    scores = [d["score"] for d in dimensions.values()]
    overall_score = sum(scores) / len(scores)
    print(f"      总体得分: {overall_score:.2f}/1.00")

    # Step 6: Generate report
    print(f"\n[6/6] 生成能力报告...")
    evaluations = {
        "dimensions": dimensions,
        "task_summary": task_summary,
        "overall_score": overall_score
    }

    generator = ReportGenerator()
    report = generator.generate(evaluations)

    report_file = REPORTS_DIR / "code_agent_capability_report.md"
    report_file.write_text(report, encoding='utf-8')

    print(f"      报告已保存: {report_file}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f" " * 22 + "Validation Complete")
    print(f"{'=' * 70}")
    print(f"\n总体得分: {overall_score:.2f}/1.00")
    print(f"能力等级: {dimensions['code_quality']['level']}")
    print(f"\n详细报告: {report_file}")
    print(f"{'=' * 70}\n")

    return evaluations


if __name__ == "__main__":
    main()
