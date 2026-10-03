# Billboard textures

`generate.mjs` turns each `assets/billboards/<name>.svg` into
`assets/billboards/<name>.msdf`, a 128x128 RGBA8 multi-channel signed distance
texture NoorRay embeds and its viewport
samples (`Viewport::createBillboardTextures`).

The SVGs are Microsoft's 24px regular Fluent System Icons
(<https://github.com/microsoft/fluentui-system-icons>, MIT). `icons.json` maps
each icon to its Fluent name; the file names are `BillboardIconNames` in
`Scene/Billboard.h`.

To change an icon, replace its SVG and run `npm install && npm run generate`
here. To add one, also add it to `BillboardIcon` and `BillboardIconNames`.
