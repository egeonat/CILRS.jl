function fix_json(f)
    fio = open(f)
    str = read(fio, String)
    close(fio)

    str = replace(str, '\'' => '"')
    str = replace(str, "False}" => "false}")
    str = replace(str, "\": nan," => "\": null,")
    str = replace(str, "True}" => "true}")

    fio = open(f, "w")
    write(fio, str)
    close(fio)
end