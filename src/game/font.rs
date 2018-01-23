use std;
use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;
use freetype as ft;
use freetype::Error as FreetypeError;
use freetype::Face;
use gfx;

use models::Image;

pub struct Font {
    pub chars: HashMap<char, BitmapChar>,

    pub texture: Image<(gfx::format::R8, gfx::format::Unorm)>,
}

pub struct FontLayout<'a> {
    pub font: &'a Font,
    pub text: String,
    pub position: [f32;2],
    pub color: [f32;4],
    pub scale: f32,
}

pub type FontResult = Result<Font, FontError>;

#[derive(Debug)]
pub enum FontError {
    FreetypeError(FreetypeError),
    EmptyFont
}

impl From<FreetypeError> for FontError {
    fn from(e: FreetypeError) -> FontError { FontError::FreetypeError(e) }
}

pub struct BitmapChar {
    pub x_offset: i32,
    pub y_offset: i32,
    pub x_advance: i32,
    pub width: i32,
    pub height: i32,
    pub tex: [f32; 2],
    pub tex_width: f32,
    pub tex_height: f32,
    // This field is used only while building the texture.
    data: Option<Vec<u8>>,
}

impl Font {
    pub fn from_path(path: &str, font_size: u8, chars: Option<&[char]>) -> FontResult {
        let library = ft::Library::init()?;
        let face = library.new_face(path, 0)?;
        Self::new(face, font_size, chars)
    }
    fn new<'a>(mut face: ft::Face<'a>, font_size: u8, chars: Option<&[char]>) -> FontResult {
        use std::iter::FromIterator;
        use std::iter::repeat;
        use std::cmp::max;

        let needed_chars = chars
            .map(|sl| HashSet::from_iter(sl.iter().cloned()))
            .unwrap_or_else(|| Self::get_all_face_chars(&mut face));
        if needed_chars.is_empty() {
            return Err(FontError::EmptyFont);
        }

        face.set_pixel_sizes(font_size as u32, font_size as u32)?;

        let mut chars = HashMap::default();
        let mut sum_image_width = 0;
        let mut max_ch_width = 0;
        let mut max_ch_height = 0;
        for ch in needed_chars {
            try!(face.load_char(ch as usize, ft::face::RENDER));

            let glyph = face.glyph();
            let bitmap = glyph.bitmap();

            let ch_width = bitmap.width();
            let ch_height = bitmap.rows();

            chars.insert(ch, BitmapChar {
                x_offset: glyph.bitmap_left(),
                y_offset: font_size as i32 - glyph.bitmap_top(),
                x_advance: (glyph.advance().x >> 6) as i32,
                width: ch_width,
                height: ch_height,
                tex: [0.0, 0.0],
                tex_width: 0.0,
                tex_height: 0.0,
                data: Some(Vec::from(bitmap.buffer()))
            });
            sum_image_width += ch_width;
            max_ch_width = max(max_ch_width, ch_width);
            max_ch_height = max(max_ch_height, ch_height);
        }

        let ideal_image_size = sum_image_width * max_ch_height;
        let image_width = {
            let ideal_image_width = (ideal_image_size as f32).sqrt() as i32;
            max(max_ch_width, ideal_image_width)
        };
        let mut image = Vec::new();
        let mut chars_row = Vec::new();
        let dump_row = |image: &mut Vec<u8>, chars_row: &Vec<(i32, i32, Vec<u8>)>| {
            for i in 0..max_ch_height {
                let mut x = 0;
                for &(width, height, ref data) in chars_row {
                    if i >= height {
                        image.extend(repeat(0).take(width as usize));
                    } else {
                        let skip = i * width;
                        let line = data.iter().skip(skip as usize).take(width as usize);
                        image.extend(line.cloned());
                    };
                    x += width;
                }
                let cols_to_fill = image_width - x;
                image.extend(repeat(0).take(cols_to_fill as usize));
            }
        };

        let mut cursor_x = 0;
        let mut image_height = 0;

        for (_, ch_info) in chars.iter_mut() {
            if cursor_x + ch_info.width > image_width {
                dump_row(&mut image, &chars_row);
                chars_row.clear();
                cursor_x = 0;
                image_height += max_ch_height;
            }
            let ch_data = ch_info.data.take().unwrap();
            chars_row.push((ch_info.width, ch_info.height, ch_data));
            ch_info.tex = [cursor_x as f32, image_height as f32];
            cursor_x += ch_info.width;
        }
        dump_row(&mut image, &chars_row);
        image_height += max_ch_height;

        for (_, ch_info) in chars.iter_mut() {
            ch_info.tex[0] /= image_width as f32;
            ch_info.tex[1] /= image_height as f32;
            ch_info.tex_width = ch_info.width as f32 / image_width as f32;
            ch_info.tex_height = ch_info.height as f32 / image_height as f32;
        }
        let texture = Image {
            data: image,
            width: image_width as u16,
            height: image_height as u16,
            format: std::marker::PhantomData::<(gfx::format::R8, gfx::format::Unorm)>
        };

        Ok(Font{
            chars,
            texture
        })
    }
    pub fn get_all_face_chars<'a>(face: &mut Face<'a>) -> HashSet<char> {
        use std::char::from_u32;
        let mut result = HashSet::default();
        let mut index = 0;
        let face_ptr = face.raw_mut();
        unsafe {
            let mut code = ft::ffi::FT_Get_First_Char(face_ptr, &mut index);
            while index != 0 {
                from_u32(code as u32).map(|ch| result.insert(ch));
                code = ft::ffi::FT_Get_Next_Char(face_ptr, code, &mut index);
            }
        }
        result
    }
    pub fn layout<'a>(&'a self, text: String, position: [f32;2], color: [f32;4], scale: f32) -> FontLayout<'a>{
        FontLayout {
            font: self,
            text,
            position,
            color,
            scale
        }
    }
}

