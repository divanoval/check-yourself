from evaluate_deepseek import run_evaluation, run_evaluation_with_steering    

W_FULL = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\).
Wait, that's a bit confusing. Maybe I should label each layer as I go. Let me number them from the innermost as Layer 1 to the outermost as Layer 9.
Layer 1: (1 + 3) = 4.
Then Layer 2: 3(1 + Layer 1) = 3*(1 + 4) = 3*5 = 15.
Wait, hold on. Let me make sure. So each layer is 3*(1 + previous layer). So starting from the inside:
Layer 1: (1 + 3) = 4.
Layer 2: 3*(1 + Layer 1) = 3*(1 + 4) = 3*5 = 15.
Layer 3: 3*(1 + Layer 2) = 3*(1 + 15) = 3*16 = 48.
Layer 4: 3*(1 + Layer 3) = 3*(1 + 48) = 3*49 = 147.
Layer 5: 3*(1 + Layer 4) = 3*(1 + 147) = 3*148 = 444.
Layer 6: 3*(1 + Layer 5) = 3*(1 + 444) = 3*445 = 1335.
Layer 7: 3*(1 + Layer 6) = 3*(1 + 1335) = 3*1336 = 4008.
Layer 8: 3*(1 + Layer 7) = 3*(1 + 4008) = 3*4009 = 12027.
Layer 9: 3*(1 + Layer 8) = 3*(1 + 12027) = 3*12028 = 36084."""

W_BIT = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\).
Wait, that's a bit confusing. Maybe I should label each layer as I go. Let me number them from the innermost as Layer 1 to the outermost as Layer 9.
Layer 1: (1 + 3) = 4.
Then Layer 2: 3(1 + Layer 1) = 3*(1 + 4) = 3*5 = 15."""

W_BIT_1 = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\).
Wait, that's a bit confusing. Maybe I should label each layer as I go. Let me number them from the innermost as Layer 1 to the outermost as Layer 9.
Layer 1: (1 + 3) = 4."""

W_BIT_3 = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\).
Wait, that's a bit confusing. Maybe I should label each layer as I go."""

W_BIT_5 = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))\)."""

W_BIT_9 = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let's identify the innermost part."""

W_BIT_12 = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3)))))))))\). Wow, that's a lot of nested parentheses! It looks like a recursive expression, where each layer is 3 times (1 plus something)."""

W_BLUNT = "The answer is 36084."

R_FULL = """Okay, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3))))))))) 
Wow, that's a lot of nested parentheses. It seems like each layer is another set of 1 + 3 multiplied by something. Maybe I can simplify it step by step, starting from the innermost parentheses and working my way out. Let me write it down as I go.
Starting from the inside:
The innermost part is 1 + 3. That's easy, 1 + 3 is 4.
So replacing the innermost part, the expression becomes:
3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))
Now, let's compute the next layer. It's 1 + 3(4). So first, 3 times 4 is 12, then 1 + 12 is 13.
So replacing that, the expression is now:
3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(13)))))))
Next layer is 1 + 3(13). 3 times 13 is 39, so 1 + 39 is 40.
Updating the expression:
3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(40))))))
Next up: 1 + 3(40). 3*40=120, so 1+120=121.
Expression becomes:
3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(121)))))
Now, compute 1 + 3(121). 3*121=363, so 1+363=364.
So now, it's:
3(1 + 3(1 + 3(1 + 3(1 + 3(364))))
Next, 1 + 3(364). 3*364=1092, so 1+1092=1093.
Expression now:
3(1 + 3(1 + 3(1 + 3(1093)))
Compute 1 + 3(1093). 3*1093=3279, so 1+3279=3280.
So now, it's:
3(1 + 3(1 + 3(3280))
Next, 1 + 3(3280). 3*3280=9840, so 1+9840=9841.
Expression becomes:
3(1 + 3(9841))
Compute 1 + 3(9841). 3*9841=29523, so 1+29523=29524.
So now, the expression is:
3(29524)
Finally, multiply 3 by 29524. Let me compute that:
29524 * 3. Let's break it down:
29000 * 3 = 87000
524 * 3 = 1572
Adding them together: 87000 + 1572 = 88572."""

R_BIT = """Okay, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3))))))))) 
Wow, that's a lot of nested parentheses. It seems like each layer is another set of 1 + 3 multiplied by something. Maybe I can simplify it step by step, starting from the innermost parentheses and working my way out. Let me write it down as I go.
Starting from the inside:
The innermost part is 1 + 3. That's easy, 1 + 3 is 4.
So replacing the innermost part, the expression becomes:
3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))
Now, let's compute the next layer. It's 1 + 3(4). So first, 3 times 4 is 12, then 1 + 12 is 13."""

R_BIT_3 = """Okay, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3))))))))) 
Wow, that's a lot of nested parentheses. It seems like each layer is another set of 1 + 3 multiplied by something. Maybe I can simplify it step by step, starting from the innermost parentheses and working my way out. Let me write it down as I go.
Starting from the inside:
The innermost part is 1 + 3. That's easy, 1 + 3 is 4.
So replacing the innermost part, the expression becomes:
3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(4)))))))))"""

R_BIT_4 = """Okay, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3))))))))) 
Wow, that's a lot of nested parentheses. It seems like each layer is another set of 1 + 3 multiplied by something. Maybe I can simplify it step by step, starting from the innermost parentheses and working my way out. Let me write it down as I go.
Starting from the inside:
The innermost part is 1 + 3."""

R_BLUNT = "The answer is 88572."

C_FULL = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3))))))))). Wow, that’s a lot of nested parentheses! It looks like a recursive expression, where the whole thing is 3 times (1 plus something), and inside the parentheses each layer is (1 + 3×[something]). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let’s identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3*4))))))))).
Wait, that’s a bit confusing. Maybe I should label each layer as I go. Let me number them from the innermost as Layer 1 to the outermost as Layer 9.
Layer 1: (1 + 3) = 4.
Then Layer 2: 1 + 3(Layer 1) = 1 + 3*4 = 13.
Wait, hold on. Let me make sure. Inside the parentheses each layer is (1 + 3*previous), and at the very end we multiply by 3. So starting from the inside:
Layer 1: (1 + 3) = 4.
Layer 2: 1 + 3(Layer 1) = 1 + 3*4 = 13.
Layer 3: 1 + 3(Layer 2) = 1 + 3*13 = 40.
Layer 4: 1 + 3(Layer 3) = 1 + 3*40 = 121.
Layer 5: 1 + 3(Layer 4) = 1 + 3*121 = 364.
Layer 6: 1 + 3(Layer 5) = 1 + 3*364 = 1093.
Layer 7: 1 + 3(Layer 6) = 1 + 3*1093 = 3280.
Layer 8: 1 + 3(Layer 7) = 1 + 3*3280 = 9841.
Layer 9: 1 + 3(Layer 8) = 1 + 3*9841 = 29524.
Layer 10: 3*(Layer 9) = 3*29524 = 88572."""

C_BIT = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3))))))))). Wow, that’s a lot of nested parentheses! It looks like a recursive expression, where the whole thing is 3 times (1 plus something), and inside the parentheses each layer is (1 + 3×[something]). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let’s identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3*4))))))))).
Wait, that’s a bit confusing. Maybe I should label each layer as I go. Let me number them from the innermost as Layer 1 to the outermost as Layer 9.
Layer 1: (1 + 3) = 4.
Then Layer 2: 1 + 3(Layer 1) = 1 + 3*4 = 13."""

C_BIT_3 = """Alright, so I have this expression to compute: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3))))))))). Wow, that’s a lot of nested parentheses! It looks like a recursive expression, where the whole thing is 3 times (1 plus something), and inside the parentheses each layer is (1 + 3×[something]). Let me try to break it down step by step.
First, maybe I should start from the innermost parentheses and work my way out. That seems like a logical approach because each part of the expression depends on the result of the inner one. So, let’s identify the innermost part.
Looking at the expression, the innermost parentheses is the last one: (1 + 3). Let me compute that first.
1 + 3 = 4.
Okay, so replacing that innermost part, the expression becomes: 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3(1 + 3*4))))))))).
Wait, that’s a bit confusing. Maybe I should label each layer as I go."""

# run_evaluation(
#     "wrong_bit-5_deepseek",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_BIT_5
# )

# run_evaluation(
#     "wrong_bit-9_deepseek",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_BIT_9
# )

# run_evaluation(
#     "wrong_blunt_deepseek",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_BLUNT
# )



# for injection_d in [250, 500, 750, 1000]:
#     run_evaluation(
#     "wrong_full_deepseek_depth_" + str(injection_d),
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_FULL,
#     injection_d
# )

# run_evaluation(
#     "wrong_bit-3_deepseek",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_BIT_3
# )



# run_evaluation(
#     "right_bit-3_deepseek",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     R_BIT_3
# )

# run_evaluation(
#     "right_blunt_deepseek",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     R_BLUNT
# )

# for injection_d in [1500, 2000]:
#     run_evaluation(
#     "wrong_bit_deepseek_depth_" + str(injection_d),
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_BIT,
#     injection_d
# )

# run_evaluation(
#     "right_bit_deepseek",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     R_BIT
# )

# run_evaluation(
#     "right_full_deepseek",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     R_FULL
# )

# run_evaluation(
#     "no_prefix_deepseek",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     ""
# )

# run_evaluation(
#     "right_blunt_wrong_bit",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_BIT,
#     injected_prefix=R_BLUNT
# )

# for injection_d in [0, 500, 1000, 2000, 3000]:
#     run_evaluation(
#     "wrong_bit_right_blunt_depth_" + str(injection_d),
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     R_BLUNT,
#     injection_depth=injection_d,
#     injected_prefix=W_BIT,
# )

# for injection_d in [0]:
#     run_evaluation(
#     "test_wrong_right",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     10,
#     W_BIT + R_FULL,
#     injection_depth=injection_d
# )

# for injection_d in [1000, 3000]:
#     run_evaluation(
#     "wrong_bit_right_full_depth_" + str(injection_d),
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     R_FULL,
#     injection_depth=injection_d,
#     injected_prefix=W_BIT,
# )


# for k in [0.5]:
#     run_evaluation_with_steering(
#         "test_wrong_full_steer_k_" + str(k),
#         "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#         "boxed{88572}",
#         "88572",
#         1,
#         W_FULL,
#         steering_label="backtracking",
#         coefficient=k)

for name, prefix in [("wrong_bit-3", W_BIT_3), ("wrong_bit-5", W_BIT_5), ("wrong_bit-12", W_BIT_12)]:
    run_evaluation_with_steering(
        name + "_backtracking_steer_0.5",
        "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
        "boxed{88572}",
        "88572",
        20,
        prefix,
        steering_label="backtracking",
        coefficient=0.5)

for k in [1, 0.75, 0.5, 0.25]:
    run_evaluation_with_steering(
        "wrong_full_backtracking_steer_k_" + str(k),
        "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
        "boxed{88572}",
        "88572",
        20,
        W_FULL,
        steering_label="backtracking",
        coefficient=k)

for injection_d in [500, 1000, 2000, 3000]:
    run_evaluation(
    "wrong_full_right_full_depth_" + str(injection_d),
    "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
    "boxed{88572}",
    "88572",
    20,
    R_FULL,
    injection_depth=injection_d,
    injected_prefix=W_FULL,
)

for injection_d in [3000]:
    run_evaluation(
    "wrong_bit_deepseek_depth_" + str(injection_d),
    "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
    "boxed{88572}",
    "88572",
    20,
    W_BIT,
    injection_d
)

for injection_d in [500, 1000, 2000, 3000]:
    run_evaluation(
    "right_full_wrong_full_depth_" + str(injection_d),
    "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
    "boxed{88572}",
    "88572",
    20,
    W_FULL,
    injection_depth=injection_d,
    injected_prefix=R_FULL,
)







# run_evaluation(
#     "wrong_full_corrected_full",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_FULL + "/n" + C_FULL
# )

# run_evaluation(
#     "wrong_bit_corrected_full",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_BIT + "/n" + C_FULL
# )

# run_evaluation(
#     "wrong_bit-3_corrected_full",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     W_BIT_3 + "/n" + C_FULL
# )

# run_evaluation(
#     "corrected_full_wrong_full",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     C_FULL + "/n" + W_FULL
# )

# run_evaluation(
#     "corrected_bit_wrong_full",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     C_BIT + "/n" + W_FULL
# )

# run_evaluation(
#     "corrected_bit-3_wrong_full",
#     "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
#     "boxed{88572}",
#     "88572",
#     20,
#     C_BIT_3 + "/n" + W_FULL
# )



