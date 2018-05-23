import sys
import zipfile
import requests


def download_dataset(url, name):
    with open(name, "wb") as dst:
        msg = "Preparing to download file '{}'".format(url.split("/")[-1])
        print(msg)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            dst.write(response.content)
        else:
            download_lenght = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                download_lenght += len(data)
                dst.write(data)
                done = int(50 * download_lenght / total_length)
                sys.stdout.write("\r[%s%s]" % ('*' * done, ' ' * (50-done)) )
                sys.stdout.flush()


def unzip_dataset(path_to_zip, dst=""):
    try:
        with zipfile.ZipFile(path_to_zip, "r") as extractor:
            extractor.extractall(dst)
        extractor.close()
        return dst
    except Exception as err:
        msg = ("Couldn't extract file '{}'. "
               "The error caught was: '{}'" ).format(path_to_zip, err)
        print(msg)
        return None
