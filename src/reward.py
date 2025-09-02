import re
from typing import Optional, List, Union, Dict, Any
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig


class Reward:
    """
    A static class for calculating various types of rewards for VLM completions.
    
    This class provides methods to evaluate the quality of model outputs based on
    format compliance and mathematical accuracy.
    """

    @staticmethod
    def format_reward(completions: List[str], **kwargs: Any) -> List[float]:
        """
        Calculate format-based rewards for completions.
        
        Evaluates whether completions follow the expected format with <think> and <answer> tags.
        
        Args:
            completions: List of completion strings to evaluate
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            list[float]: List of reward values where 1.0 indicates correct format and 0.0 indicates incorrect format
            
        Example:
            >>> Reward.format_reward(["<think>reasoning</think>\\n<answer>solution</answer>"])
            [1.0]
        """
        pattern: str = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        matches: List[Optional[re.Match]] = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        rewards: List[float] = [1.0 if match else 0.0 for match in matches]
        return rewards

    @staticmethod
    def accuracy_reward(completions: List[Union[str, List[Dict[str, str]]]], solution: List[str], **kwargs: Any) -> List[Optional[float]]:
        """
        Calculate accuracy-based rewards for completions against solutions.
        
        Evaluates mathematical accuracy by parsing LaTeX expressions and verifying
        mathematical equivalence. Falls back to string comparison for non-LaTeX content.
        
        Args:
            completions: List of completion data structures or strings
            solution: List of solution strings to compare against
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            list[Optional[float]]: List of reward values between 0.0 and 1.0, 
                                  or None if verification fails
                                  
        Note:
            - For LaTeX content: Uses math-verify to parse and verify mathematical equivalence
            - For non-LaTeX content: Performs case-insensitive string comparison
            - Returns None when mathematical verification fails due to parsing errors
        """
        rewards: List[Optional[float]] = []

        for completion, sol in zip(completions, solution):
            try:
                gold_parsed: List = parse(sol, extraction_mode="first_match") # check if it's latex-format mathematical representation
            except Exception as e:
                gold_parsed: List = []

            if len(gold_parsed) != 0:
                try:
                    answer_parsed: List = parse(
                        completion,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed="all",
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                    reward: float = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                    reward: Optional[float] = None
            else:
                # Handle case where completion might be a string or a list
                completion_text: str = completion if isinstance(completion, str) else str(completion)
                reward: float = float(completion_text.strip().lower() == sol.strip().lower())

            rewards.append(reward)

        return rewards