#!/usr/bin/env python3
"""Cleanup SageMaker experiments/trials/components/endpoints with optional dry-run."""
from __future__ import annotations

import argparse
import boto3


def delete_experiments(sagemaker, *, dry_run: bool) -> None:
    paginator = sagemaker.get_paginator("list_experiments")
    for page in paginator.paginate():
        for exp in page.get("ExperimentSummaries", []):
            name = exp["ExperimentName"]
            print(f"[experiment] delete {name} (dry_run={dry_run})")
            if dry_run:
                continue
            sagemaker.delete_experiment(ExperimentName=name, DeleteChildren=True)


def delete_endpoints(sagemaker, *, dry_run: bool) -> None:
    paginator = sagemaker.get_paginator("list_endpoints")
    for page in paginator.paginate():
        for ep in page.get("Endpoints", []):
            ep_name = ep["EndpointName"]
            ep_status = ep.get("EndpointStatus")
            print(f"[endpoint] delete {ep_name} status={ep_status} (dry_run={dry_run})")
            if dry_run:
                continue
            try:
                desc = sagemaker.describe_endpoint(EndpointName=ep_name)
                cfg = desc.get("EndpointConfigName")
                sagemaker.delete_endpoint(EndpointName=ep_name)
                if cfg:
                    sagemaker.delete_endpoint_config(EndpointConfigName=cfg)
            except Exception as exc:  # pragma: no cover
                print(f"  ! failed to delete {ep_name}: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delete SageMaker experiments/endpoints")
    p.add_argument("--region", default=None)
    p.add_argument("--dry-run", action="store_true", help="Print actions without deleting")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session_kwargs = {}
    if args.region:
        session_kwargs["region_name"] = args.region
    sm = boto3.client("sagemaker", **session_kwargs)

    delete_experiments(sm, dry_run=args.dry_run)
    delete_endpoints(sm, dry_run=args.dry_run)

    print("done")


if __name__ == "__main__":
    main()
