"""Tests for remove_imports utility function."""

from agents.ds_star.ds_star_utils import remove_imports


class TestRemoveImports:
    """Test import removal logic in remove_imports function."""

    def test_removes_single_line_imports(self):
        """Test that single-line import statements are removed."""
        code_with_imports = """import numpy as np
import pandas as pd
from math import sqrt

result = np.array([1, 2, 3])
print(result)"""

        cleaned = remove_imports(code_with_imports)

        assert "import numpy" not in cleaned
        assert "import pandas" not in cleaned
        assert "from math import" not in cleaned
        assert "result = np.array" in cleaned
        assert "print(result)" in cleaned

    def test_removes_multiline_imports(self):
        """Test that multiline imports are removed."""
        code_with_imports = """import numpy as np
from typing import (
    List,
    Dict
)

data = [1, 2, 3]"""

        cleaned = remove_imports(code_with_imports)

        assert "import numpy" not in cleaned
        assert "from typing" not in cleaned
        assert "List" not in cleaned
        assert "Dict" not in cleaned
        assert "data = [1, 2, 3]" in cleaned

    def test_preserves_non_import_code(self):
        """Test that non-import code is preserved."""
        code_without_imports = """# This is a comment
result = call_tool('test_tool', query='test')
print(result)"""

        cleaned = remove_imports(code_without_imports)

        assert cleaned.strip() == code_without_imports.strip()

    def test_removes_imports_with_comments(self):
        """Test that imports with inline comments are removed."""
        code_with_imports = """import numpy as np  # for arrays
from pandas import DataFrame  # for dataframes

data = np.array([1, 2, 3])"""

        cleaned = remove_imports(code_with_imports)

        assert "import numpy" not in cleaned
        assert "from pandas" not in cleaned
        assert "data = np.array" in cleaned

    def test_handles_non_string_input(self):
        """Test that non-string input is converted to string."""
        # Test with None
        result = remove_imports(None)
        assert result == ""

        # Test with number (edge case)
        result = remove_imports(123)
        assert result == "123"

    def test_handles_invalid_syntax(self):
        """Test fallback for invalid Python syntax."""
        invalid_code = """import numpy
this is not valid python
from pandas import"""

        # Should not raise exception, uses fallback regex approach
        cleaned = remove_imports(invalid_code)
        assert "import numpy" not in cleaned
        assert "from pandas" not in cleaned
        assert "this is not valid python" in cleaned

    def test_preserves_import_in_strings(self):
        """Test that 'import' in strings is preserved."""
        code = """message = "import this module"
print(message)"""

        cleaned = remove_imports(code)
        assert "import this module" in cleaned
        assert "print(message)" in cleaned

    def test_removes_nested_imports_in_functions(self):
        """Test that imports inside functions are removed."""
        code = """def my_function():
    import numpy as np
    import pandas as pd
    result = np.array([1, 2, 3])
    return result

# This is a comment
x = 5"""

        cleaned = remove_imports(code)

        # Imports should be removed
        assert "import numpy" not in cleaned
        assert "import pandas" not in cleaned

        # Function structure and other code should be preserved
        assert "def my_function():" in cleaned
        assert "result = np.array([1, 2, 3])" in cleaned
        assert "return result" in cleaned
        assert "# This is a comment" in cleaned
        assert "x = 5" in cleaned

    def test_removes_nested_imports_in_conditionals(self):
        """Test that imports inside conditionals are removed."""
        code = """if True:
    import sys
    print(sys.version)
else:
    from os import path
    print(path.exists('.'))"""

        cleaned = remove_imports(code)

        # Imports should be removed
        assert "import sys" not in cleaned
        assert "from os import" not in cleaned

        # Conditional structure and other code should be preserved
        assert "if True:" in cleaned
        assert "print(sys.version)" in cleaned
        assert "else:" in cleaned
        assert "print(path.exists('.'))" in cleaned
