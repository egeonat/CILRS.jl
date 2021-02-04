using ImageView

# Function used to fix faulty json formats in LBC dataset used in previous versions
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

count = 0

# Visualize rgb images given by Carla100Data structs
function visualize_rgb(rgb)
    copy_rgb = Array{}(copy(rgb))
    println(summary(rgb))
    println(size(copy_rgb, 1))
    println(size(copy_rgb, 2))
    println(size(copy_rgb, 3))
    if ndims(rgb) == 4
        copy_rgb = reshape(copy_rgb, (size(copy_rgb, 1), size(copy_rgb, 2), size(copy_rgb, 3)))
    end
    copy_rgb = colorview(RGB, permutedims(copy_rgb, (3, 1, 2)))
    imshow(copy_rgb, name=string("RGB - ", count))
    global count += 1
end
