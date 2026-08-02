-- In-place `a[i] = a[i] * scale + offset`. Matcha's mel denormalization is the affine case, VITS's
-- pre-scaled duration noise the `offset = 0` one; naming it keeps both from being a bare loop whose
-- purpose is only recoverable from the surrounding comment.
local function array_affine(a, scale, offset)
    for i = 1, #a do a[i] = a[i] * scale + offset end
    return a
end
