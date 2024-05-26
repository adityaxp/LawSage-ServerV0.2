from datetime import datetime


def write_log(message, service):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(f'LawSage v0.3-{service}-[{formatted_datetime}]> '+ str(message))
