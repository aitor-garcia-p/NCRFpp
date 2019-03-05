import tempfile

# with tempfile.NamedTemporaryFile(encoding='utf-8', delete=True) as f:
#     f.write('asdasdasdasd')
#
#     aa = f.read()
#
#     print(aa)


fp = tempfile.NamedTemporaryFile()
fp.write(b'Hello world!')
fp.seek(0)
content = fp.read()
print(content)
fp.close()


with tempfile.NamedTemporaryFile() as fp:
    fp.write(b'asdasdasdasd')
    fp.seek(0)
    aa = fp.read()

    print(aa)