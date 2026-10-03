// Turns every assets/billboards/*.svg into a multi-channel signed distance
// texture next to it: TEXTURE_SIZE squared RGBA8 texels, top row first, with no
// header, which the viewport uploads as it is. msdfgen only reads fonts, so the icons are packed into a
// throwaway font whose glyphs are the icon outlines.
//
// The constants must match ViewportBillboardTextureSize and
// ViewportBillboardDistanceRange in Shared/Viewport.h.
import { readFileSync, writeFileSync, readdirSync } from "node:fs"
import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import path from "node:path"

const require = createRequire(import.meta.url)
const opentype = require("opentype.js")
const svgpath = require("svgpath")
const { Msdfgen } = require("msdfgen-wasm")

const TEXTURE_SIZE = 128
const DISTANCE_RANGE_PIXELS = 12
// Clear border around the 24 unit icon grid; it has to hold half the range.
const MARGIN_PIXELS = 8
const GRID = 24
const UNITS_PER_EM = 1000
const FIRST_CODE_POINT = 0xe000

const assets = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../assets/billboards")
const names = readdirSync(assets).filter(file => file.endsWith(".svg")).map(file => file.slice(0, -4)).sort()

function glyphPath(svg) {
    const pathData = [...svg.matchAll(/<path[^>]* d="([^"]+)"/g)].map(match => match[1]).join(" ")
    const scale = UNITS_PER_EM / GRID
    const x = value => value * scale
    // SVG grows downwards, fonts upwards.
    const y = value => (GRID - value) * scale
    const outline = new opentype.Path()
    svgpath(pathData).abs().unarc().unshort().iterate((segment, _index, startX, startY) => {
        const [command, ...v] = segment
        switch (command) {
            case "M": outline.moveTo(x(v[0]), y(v[1])); break
            case "L": outline.lineTo(x(v[0]), y(v[1])); break
            case "H": outline.lineTo(x(v[0]), y(startY)); break
            case "V": outline.lineTo(x(startX), y(v[0])); break
            case "Q": outline.quadraticCurveTo(x(v[0]), y(v[1]), x(v[2]), y(v[3])); break
            case "C": outline.curveTo(x(v[0]), y(v[1]), x(v[2]), y(v[3]), x(v[4]), y(v[5])); break
            case "Z": outline.close(); break
            default: throw new Error(`unsupported path command ${command}`)
        }
    })
    return outline
}

const glyphs = [new opentype.Glyph({ name: ".notdef", unicode: 0, advanceWidth: UNITS_PER_EM, path: new opentype.Path() })]
names.forEach((name, index) => glyphs.push(new opentype.Glyph({
    name, unicode: FIRST_CODE_POINT + index, advanceWidth: UNITS_PER_EM,
    path: glyphPath(readFileSync(path.join(assets, `${name}.svg`), "utf8"))
})))
const font = new opentype.Font({
    familyName: "Billboards", styleName: "Regular", unitsPerEm: UNITS_PER_EM,
    ascender: UNITS_PER_EM, descender: 0, glyphs
})

const msdfgen = await Msdfgen.create(readFileSync(require.resolve("msdfgen-wasm/wasm")))
msdfgen.loadFont(new Uint8Array(font.toArrayBuffer()))
msdfgen.loadGlyphs(names.map((_, index) => FIRST_CODE_POINT + index), { preprocess: true })

// One em is the whole 24 unit grid; the glyph origin is the grid's corner.
const scale = TEXTURE_SIZE - 2 * MARGIN_PIXELS
names.forEach((name, index) => {
    const glyph = msdfgen.getGlyph(FIRST_CODE_POINT + index)
    const bitmap = msdfgen.generateBitmap(glyph, {
        width: TEXTURE_SIZE, height: TEXTURE_SIZE, scale,
        range: DISTANCE_RANGE_PIXELS / scale,
        xTranslate: MARGIN_PIXELS / scale, yTranslate: MARGIN_PIXELS / scale,
        edgeColoring: "simple", edgeThresholdAngle: 3, scanline: false
    })
    writeFileSync(path.join(assets, `${name}.msdf`), Buffer.from(bitmap.buffer))
})
console.log(`wrote ${names.length} textures`)
