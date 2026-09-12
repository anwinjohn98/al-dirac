from __future__ import annotations

import ast
from typing import Any


class ScoreExpressionEvaluator:
    def __init__(
        self,
        *,
        score_expression: str,
    ) -> None:
        self.score_expression = score_expression

    def _eval_node(self, node: ast.AST, record: dict[str, Any]) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.Name):
            value = record.get(node.id)
            if value is None:
                raise ValueError(f"Missing score field: {node.id}")
            return float(value)

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, record)
            right = self._eval_node(node.right, record)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right

            raise ValueError("Only +, -, *, / are allowed.")

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, record)

            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand

            raise ValueError("Only unary + and - are allowed.")

        raise ValueError("Unsupported expression.")

    def evaluate(self, record: dict[str, Any]) -> float | None:
        try:
            tree = ast.parse(self.score_expression, mode="eval")
            return float(self._eval_node(tree.body, record))
        except Exception:
            return None
