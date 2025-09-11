from evaluate_deepseek import run_evaluation, run_evaluation_with_steering    
from all_prefixes import W_S28, W_S16, W_S15, W_S13, W_S11, W_S7, W_S3, W_BLUNT, C_S28

### 1) SABOTAGE WITH BAD BEGINNINGS (VARY LENGTH) ###

    #### A) Without steering ####

for name, prefix in [("wrong_s28", W_S28), 
                    ("wrong_s16", W_S16), 
                    ("wrong_s15", W_S15), 
                    ("wrong_s13", W_S13), 
                    ("wrong_s11", W_S11), 
                    ("wrong_s7", W_S7), 
                    ("wrong_s3", W_S3), 
                    ("wrong_blunt", W_BLUNT), 
                    ("no_prefix", "")]:
    run_evaluation(
        name,
        "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
        "boxed{88572}",
        "88572",
        20,
        prefix)

    #### B) With steering ####

for name, prefix in [("wrong_s28", W_S28), 
                    ("wrong_s16", W_S16), 
                    ("wrong_s15", W_S15), 
                    ("wrong_s13", W_S13), 
                    ("wrong_s11", W_S11), 
                    ("wrong_s7", W_S7), 
                    ("wrong_s3", W_S3), 
                    ("wrong_blunt", W_BLUNT), 
                    ("no_prefix", "")]:
    run_evaluation_with_steering(
        name + "_steer",
        "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
        "boxed{88572}",
        "88572",
        20,
        prefix,
        steering_label="backtracking",
        coefficient=0.5)

### 2) MISLEAD WITH FLAWED INJECTIONS (VARY DEPTH) ###

    #### A) Full flawed prefix ####

for depth in [0, 250, 500, 750, 1000, 1500, 2000, 3000]:
    run_evaluation(
        "wrong_s28_depth_" + str(depth),
        "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
        "boxed{88572}",
        "88572",
        20,
        W_S28,
        injection_depth=depth)

    #### B) Partial flawed prefix ####

for depth in [0, 250, 500, 750, 1000, 1500, 2000, 3000]:
    run_evaluation(
        "wrong_s16_depth_" + str(depth),
        "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
        "boxed{88572}",
        "88572",
        20,
        W_S16,
        injection_depth=depth)

### 3) REDIRECT WITH CONTRARY INJECTIONS (VARY DEPTH) ###

    #### A) Correct flawed thinking ####

for depth in [0, 500, 1000, 2000, 3000]:
    run_evaluation(
        "steer_wrong_with_corrected_depth_" + str(depth),
        "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
        "boxed{88572}",
        "88572",
        20,
        C_S28,
        injection_depth=depth,
        injected_prefix=W_S28)

    #### B) Derail correct thinking ####

for depth in [0, 500, 1000, 2000, 3000]:
    run_evaluation(
        "steer_corrected_with_wrong_depth_" + str(depth),
        "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
        "boxed{88572}",
        "88572",
        20,
        W_S28,
        injection_depth=depth,
        injected_prefix=C_S28)