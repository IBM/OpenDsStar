"""
Execution environment for analyzer agent.
This module reuses the DS*Star execution environment.
"""

from agents.ds_star.ds_star_execute_env import execute_user_code

# Re-export for backward compatibility
__all__ = ["execute_user_code"]
