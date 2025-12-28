import csv
from typing import Iterator


def read_csv_file(filename: str, delimiter: str) -> list[dict[str, str]]:
    '''
    Reads .csv file and returns row-major list.
    
    :param filename: .csv file to read from
    :type filename: str
    :param delimiter: delimiter of .csv file
    :type delimiter: str
    :return: File contents as dict.
    :rtype: Iterator[dict[str, str]]
    '''

    with open(filename, 'r') as file:
        content = csv.DictReader(file, delimiter=delimiter)

        # Don't want to have to keep file open
        content = list(content)
    
    return content