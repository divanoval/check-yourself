#!/usr/bin/env python3
"""
to_multiline.py

Convert a single-line string containing the placeholder " <nEwLiNe>" into
multiline text by replacing each occurrence with a CRLF ("\r\n").

Usage examples:
  - From argument, print to stdout:
      python to_multiline.py "Hello <nEwLiNe>World"

  - From argument, write to file:
      python to_multiline.py "Hello <nEwLiNe>World" -o output.txt

  - From stdin (end with Ctrl-D), write to stdout:
      echo "Line1 <nEwLiNe>Line2" | python to_multiline.py
"""

import argparse
import sys
from typing import Optional


PLACEHOLDER = " <nEwLiNe>"

# Optional built-in defaults so you can just run the script without args.
# Set DEFAULT_INPUT to the single-line text to convert, and DEFAULT_OUTPUT
# to a filename to write. Leave either as None to fall back to CLI/stdin/stdout.
DEFAULT_INPUT: Optional[str] = "Okay, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))  <nEwLiNe>Wow, that's a lot of nested parentheses. It seems like each layer is another set of 1 + 3 multiplied by something. Maybe I can simplify it step by step, starting from the innermost parentheses and working my way out. Let me write it down as I go. <nEwLiNe>Starting from the inside: <nEwLiNe>The innermost part is 1 + 3. That's easy, 1 + 3 is 4. <nEwLiNe>So replacing the innermost part, the expression becomes: <nEwLiNe>3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4))))))))) <nEwLiNe>Now, let's compute the next layer. It's 1 + 3(4). So first, 3 times 4 is 12, then 1 + 12 is 13. <nEwLiNe>So replacing that, the expression is now: <nEwLiNe>3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(13))))))) <nEwLiNe>Next layer is 1 + 3(13). 3 times 13 is 39, so 1 + 39 is 40. <nEwLiNe>Updating the expression: <nEwLiNe>3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(40)))))) <nEwLiNe>Next up: 1 + 3(40). 3*40=120, so 1+120=121. <nEwLiNe>Expression becomes: <nEwLiNe>3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(121))))) <nEwLiNe>Now, compute 1 + 3(121). 3*121=363, so 1+363=364. <nEwLiNe>So now, it's: <nEwLiNe>3(1 + 3(1 + 3(1 + 3(1 + 3(364)))) <nEwLiNe>Next, 1 + 3(364). 3*364=1092, so 1+1092=1093. <nEwLiNe>Expression now: <nEwLiNe>3(1 + 3(1 + 3(1 + 3(1093))) <nEwLiNe>Compute 1 + 3(1093). 3*1093=3279, so 1+3279=3280. <nEwLiNe>So now, it's: <nEwLiNe>3(1 + 3(1 + 3(3280)) <nEwLiNe>Next, 1 + 3(3280). 3*3280=9840, so 1+9840=9841. <nEwLiNe>Expression becomes: <nEwLiNe>3(1 + 3(9841)) <nEwLiNe>Compute 1 + 3(9841). 3*9841=29523, so 1+29523=29524. <nEwLiNe>So now, the expression is: <nEwLiNe>3(29524) <nEwLiNe>Finally, multiply 3 by 29524. Let me compute that: <nEwLiNe>29524 * 3. Let's break it down: <nEwLiNe>29000 * 3 = 87000 <nEwLiNe>524 * 3 = 1572 <nEwLiNe>Adding them together: 87000 + 1572 = 88572 <nEwLiNe>So, putting it all together, each step simplifies the expression, starting from the innermost parentheses and moving outward. Each time, I compute the innermost expression, replace it, and then move outward until I reach the final result. <nEwLiNe>Wait, let me double-check my calculations to make sure I didn't make a mistake somewhere. <nEwLiNe>Starting from the innermost: <nEwLiNe>1 + 3 = 4 <nEwLiNe>Then, 1 + 3*4 = 1 + 12 = 13 <nEwLiNe>Next, 1 + 3*13 = 1 + 39 = 40 <nEwLiNe>Then, 1 + 3*40 = 1 + 120 = 121 <nEwLiNe>Next, 1 + 3*121 = 1 + 363 = 364 <nEwLiNe>Then, 1 + 3*364 = 1 + 1092 = 1093 <nEwLiNe>Next, 1 + 3*1093 = 1 + 3279 = 3280 <nEwLiNe>Then, 1 + 3*3280 = 1 + 9840 = 9841 <nEwLiNe>Next, 1 + 3*9841 = 1 + 29523 = 29524 <nEwLiNe>Finally, 3*29524 = 88572 <nEwLiNe>Hmm, that seems consistent each time. So, I think my calculations are correct. Each step builds on the previous one, multiplying by 3 and adding 1 each time. It's interesting how the numbers grow exponentially as we move outward. <nEwLiNe>So, after carefully simplifying each layer step by step, the final result is 88,572. <nEwLiNe>**Final Answer** <nEwLiNe>The value of the expression is \boxed{88572}. <nEwLiNe>"

DEFAULT_OUTPUT: Optional[str] = "/workspace/check-yourself/parsed_outputs/88572.txt"


def convert_placeholder_to_crlf(text: str) -> str:
    """Replace placeholder tokens with CRLF line breaks."""
    return text.replace(PLACEHOLDER, "\r\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert placeholder tokens to CRLF line breaks.")
    parser.add_argument("text", nargs="?", help="Input text. If omitted, read from stdin.")
    parser.add_argument("-o", "--output", help="Output file path. If omitted, write to stdout.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if args.text is not None:
        input_text = args.text
    else:
        if DEFAULT_INPUT is not None:
            input_text = DEFAULT_INPUT
        else:
            input_text = sys.stdin.read()

    converted = convert_placeholder_to_crlf(input_text)

    output_path = args.output if args.output else DEFAULT_OUTPUT
    if output_path:
        # Write as-is; converted contains explicit \r\n line breaks
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(converted)
    else:
        sys.stdout.write(converted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


