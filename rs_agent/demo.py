from __future__ import annotations

import argparse
import json

from rs_agent.factory import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the remote-sensing assistant MVP demo.")
    parser.add_argument("--data-root", default=".rs_agent_data")
    parser.add_argument("--auto-confirm", action="store_true", help="Skip plan review and execute directly.")
    args = parser.parse_args()

    runtime = build_runtime(args.data_root)
    state = runtime.create_task(
        user_goal="帮我对两期 Sentinel-2 影像做建设用地扩张变化检测，输出图斑、面积统计和报告。",
        image_t1="demo://image_t1",
        image_t2="demo://image_t2",
        auto_confirm=args.auto_confirm,
    )
    if state.status == "waiting_human":
        interrupt = state.interrupts[-1]
        print("Task is waiting for plan confirmation.")
        print(json.dumps({"task_id": state.task_id, "interrupt_id": interrupt.interrupt_id}, ensure_ascii=False, indent=2))
        print("Approve with the API, or rerun with --auto-confirm for an end-to-end demo.")
        return

    print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

