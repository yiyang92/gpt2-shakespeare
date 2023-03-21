from subprocess import call


def main() -> None:
    call("uvicorn gpt_2_shakespeare_service.service:app --reload", shell=True)
