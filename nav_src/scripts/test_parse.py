import re
import unittest

def extract_action_and_tool_input(text):
    regex = r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*\"?([a-fA-F0-9]{32})\"?"

    action_match = re.search(regex, text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        tool_input = action_match.group(2).strip()
        return action, tool_input
    else:
        return None, None

class TestActionAndToolInputExtraction(unittest.TestCase):

    def test_extraction(self):
        samples = [
            ("which tells me ... Action: action_maker\nAction Input: \"f237319a500640d8ac172db225a3ce9c\" (Left viewpoint ID)", "action_maker", "f237319a500640d8ac172db225a3ce9c"),
            ("which is to turn right ... Action: action_maker\nAction Input: \"06bd0a2d004b454b9e93ddcf08344732\"", "action_maker", "06bd0a2d004b454b9e93ddcf08344732"),
            ("which is to exit out ... Action: action_maker\nAction Input: \"424bcb744623413f830ece5c68319d70\"\n", "action_maker", "424bcb744623413f830ece5c68319d70")
        ]

        for idx, (sample, expected_action, expected_tool_input) in enumerate(samples, 1):
            action, tool_input = extract_action_and_tool_input(sample)
            
            # Print statements
            print(f"Testing Sample {idx} ...")
            print(f"Expected Action: {expected_action}, Output: {action}")
            print(f"Expected Tool Input: {expected_tool_input}, Output: {tool_input}\n")
            
            self.assertEqual(action, expected_action)
            self.assertEqual(tool_input, expected_tool_input)

if __name__ == '__main__':
    unittest.main()
