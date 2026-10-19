"""Portable CLI for the standalone exact Wayback snapshot fetcher."""
import argparse
import json
from wayback_snapshot_fetcher import Crawler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='Frozen JSON input manifest')
    parser.add_argument('--output',required=True,help='Private output directory; safe to resume')
    parser.add_argument('--concurrency',type=int,choices=(1,2),default=1)
    parser.add_argument('--timeout',type=float,default=20)
    parser.add_argument('--max-attempts',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--backoff-base',type=float,default=30)
    parser.add_argument('--backoff-cap',type=float,default=300)
    args=parser.parse_args()
    crawler=Crawler(args.output,timeout=args.timeout,concurrency=args.concurrency,
                    max_attempts=args.max_attempts,backoff_base=args.backoff_base,backoff_cap=args.backoff_cap)
    report=crawler.run(args.input)
    print(json.dumps({k:v for k,v in report.items() if k not in {'files','attempt_results','successful_responses'}},indent=2))


if __name__=='__main__':main()
