import argparse
import logging
import subprocess
import sys

from dotenv import load_dotenv


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help="Entry points")
    args = parser.parse_args()

    return args


def build_command(args):
    command = ["mlflow", "run"]
    if args.e:
        command = [*command, "-e", args.e]
    url = "."
    command.append(url)
    return command


set_variables = load_dotenv(override=True)
if not set_variables:
    logging.warning("No environment variables have been set.")

args = get_cli_args()
command = build_command(args)
subprocess.run(command, stderr=sys.stderr, stdout=sys.stdout)

# TODO Check if docker is up and running. If not exit with a verbose error message
# subprocess.run(["docker", "build", "-t", "mlflow-docker-example", "."], stderr=sys.stderr, stdout=sys.stdout)
# print("\nBuilt docker image.\nStarting mlflow run...\n")
