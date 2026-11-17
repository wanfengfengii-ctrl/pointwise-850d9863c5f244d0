"""公开事故：固定队列的模型参数与共同超限报告偏离已公开参照。
运行 python3.11 -B incident.py；正常应显示两个事故结果均满足参照。
这不是网络或依赖检查；执行异常单独退出，不输出业务故障标记。
输入、统计支持、原回归及固定成果接口见 README.md 和 samples.py。
"""
import json
import math
import sys
from lab.workflow import Workflow
from samples import exact_rows, mixed_rows, REFERENCE

def evaluate(rows, thresholds, expected):
    with Workflow() as workflow:
        workflow.put_batch("incident", rows)
        task = workflow.freeze("incident", sorted({row["cohort_id"] for row in rows}))
        fit = workflow.fit(task["task_id"])
        report = workflow.risk(fit["result_id"], thresholds)
        observed = {"mu": fit["mu"], "covariance": fit["covariance"],
                    "log_likelihood": fit["log_likelihood"],
                    "probability": report["probability"]}
        errors = {"mu": max(abs(a - b) for a, b in zip(observed["mu"], expected["mu"])),
                  "covariance": max(abs(observed["covariance"][i][j] - expected["covariance"][i][j])
                                    for i in range(2) for j in range(2)),
                  "probability": abs(observed["probability"] - expected["probability"])}
        if "log_likelihood" in expected:
            errors["log_likelihood"] = abs(observed["log_likelihood"] - expected["log_likelihood"])
        passed = errors["mu"] <= 1e-6 and errors["covariance"] <= 1e-6 and errors["probability"] <= 1e-7
        if "log_likelihood" in errors:
            passed = passed and errors["log_likelihood"] <= 1e-8
        return {"input": rows, "thresholds": thresholds, "expected": expected,
                "observed": observed, "absolute_errors": errors, "passed": passed}

def main():
    exact = {"mu": [0., 0.], "covariance": [[2/3, 1/3], [1/3, 2/3]],
             "probability": 1/3}
    cases = [("correlated_exact", exact_rows(), [0., 0.], exact),
             ("mixed_censoring", mixed_rows(), REFERENCE["thresholds"], REFERENCE)]
    results = {}
    try:
        for name, rows, thresholds, expected in cases:
            results[name] = evaluate(rows, thresholds, expected)
    except Exception as exc:
        print("INCIDENT_EXECUTION_ERROR " + json.dumps({"type": type(exc).__name__, "message": str(exc)},
                                                       ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(results, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if not all(case["passed"] for case in results.values()):
        print("JOINT_CENSOR_RISK_MISMATCH")
        return 1
    print("JOINT_CENSOR_RISK_OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
