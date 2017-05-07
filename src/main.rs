#[macro_use]
extern crate gfx;
extern crate glutin;
extern crate gfx_window_glutin;
extern crate cgmath;
extern crate rusqlite;
extern crate fnv;
extern crate coarsetime;
extern crate gfx_device_gl;

extern crate freetype;

use rusqlite::Connection;
use rusqlite::Error as RusqliteError;
use std::path::Path;
use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;

use freetype as ft;
use freetype::Error as FreetypeError;
use freetype::Face;

use gfx::memory::Typed;

pub type ColorFormat = gfx::format::Srgba8;
pub type DepthFormat = gfx::format::Depth;

use cgmath:: {
    EuclideanSpace,
    Point3,
    Vector3,
    Matrix4,
    One,
    Zero,
    // Rotation,
    // Quaternion,
    Transform,
};
type FontResult = Result<Font, FontError>;
type RusqliteResult<T> = Result<T, RusqliteError>;


#[derive(Debug)]
pub enum AppError {
    RusqliteError(RusqliteError),
    FontError(FontError)
}

#[derive(Debug)]
pub enum FontError {
    FreetypeError(FreetypeError),
    EmptyFont
}

impl From<FreetypeError> for FontError {
    fn from(e: FreetypeError) -> FontError { FontError::FreetypeError(e) }
}
impl From<RusqliteError> for AppError {
    fn from(e: RusqliteError) -> AppError { AppError::RusqliteError(e) }
}
impl From<FontError> for AppError {
    fn from(e: FontError) -> AppError { AppError::FontError(e) }
}
impl<T: gfx::format::TextureFormat> From<Font> for Image<T> {
    fn from(f: Font) -> Image<T> { Image { data: f.image, width: f.width, height: f.height , format: std::marker::PhantomData::<T>} }
}

pub fn main() {
    use gfx::Device;
    use gfx::traits::FactoryExt;
    use gfx::Factory;

    let conn = Connection::open(&Path::new("file.db")).expect("failed to open sqlite file");
    let width = 1024;
    let height = 768;

    let (window, mut device, mut factory, main_color, main_depth) = {
        let builder = glutin::WindowBuilder::new()
            .with_title("PARTI")
            .with_dimensions(width, height)
            .with_vsync();
        gfx_window_glutin::init::<ColorFormat, DepthFormat>(builder)
    };

    let pso = {
        let shaders = factory.create_shader_set(include_bytes!("shader/150.glslv"),
        include_bytes!("shader/150.glslf")).expect("shader exists?");
        factory.create_pipeline_state(
            &shaders,
            gfx::Primitive::TriangleList,
            gfx::state::Rasterizer::new_fill(),
            pipe_w::new()
            ).expect("failed to create pipeline w")
    };
    // let pso_w2 = {
    //     let shaders = factory.create_shader_set(include_bytes!("shader/150.glslv"),
    //     include_bytes!("shader/150.glslf")).expect("shader exists?");
    //     factory.create_pipeline_state(
    //         &shaders,
    //         gfx::Primitive::TriangleList,
    //         gfx::state::Rasterizer::new_fill(),
    //         pipe_w2::new()
    //         ).expect("failed to create pipeline w2")
    // };
    let pso_p = {
        let shaders = factory.create_shader_set(b"
#version 150 core

in vec3 position;
in vec4 color;
out vec4 v_color;

void main() {
    gl_Position = vec4(position, 1.0);
    v_color = color;
}
",
b"
#version 150 core
in vec4 v_color;
out vec4 Target0;

void main() {
    Target0 = v_color;
}
").expect("shader error");
        factory.create_pipeline_state(
            &shaders,
            gfx::Primitive::LineStrip,
            gfx::state::Rasterizer::new_fill().with_cull_back(),
            pipe_p::new()
        ).expect("failed to create pipeline")
    };
    
    let sampler = {
        let sampler_info = gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Trilinear,
            gfx::texture::WrapMode::Clamp);
        factory.create_sampler(sampler_info)
    };
    
    let mut encoder: gfx::Encoder<_, _> = factory.create_command_buffer().into();
    let mut entries = query_entry(&conn, &mut factory, &[1]).unwrap();
    let mut camera = Camera::new(Point3::new(30.0, -40.0, 30.0),
    Point3::new(0.0, 0.0, 0.0),
    cgmath::PerspectiveFov {
        fovy: cgmath::Rad(16.0f32.to_radians()),
        aspect: (width as f32) / (height as f32),
        near: 5.0,
        far: 1000.0,
    });
    
    let font_entry = font_entry(&mut factory, "に");
    
    'main: loop {
        let timer = coarsetime::Instant::now();
        {
            let mut avator = entries.get_mut(&1).unwrap();
            for event in window.poll_events() {
                match event_handler(event) {
                    Some(GameCommand::Exit) => break 'main,
                    Some(GameCommand::CameraMove(v)) => { camera.translate(v); },
                    Some(GameCommand::AvatorMove(v)) => { avator.translate(v); },
                    _ => {}
                }
            }
            camera.look_to(avator.position + Vector3::new(0.0, 0.0, 10.0));
            camera.update();
        }
    
        encoder.clear(&main_color.clone(), CLEAR_COLOR);
        encoder.clear_depth(&main_depth, 1.0);
    
        for obj in entries.values() {
            let mv = camera.view * Matrix4::from_translation(obj.position.to_vec());
            let mvp = camera.perspective * mv;
            let skinning_buffer = factory.create_constant_buffer::<Skinning>(64);
            encoder.update_buffer(&skinning_buffer, &obj.skinning[..], 0).unwrap();
            for entry in &obj.entries {
                let data = pipe_w::Data {
                    vbuf: entry.vertex_buffer.clone(),
                    u_model_view_proj: mvp.into(),
                    u_model_view: mv.into(),
                    u_light: [0.2, 0.2, -0.2f32],
                    u_ambient_color: [0.01, 0.01, 0.01, 1.0],
                    u_eye_direction: camera.direction().into(),
                    u_texture: (entry.texture.clone(), sampler.clone()),
                    out_color: main_color.clone(),
                    out_depth: main_depth.clone(),
                    b_skinning: skinning_buffer.raw().clone(),
                };
                encoder.draw(&entry.slice, &pso, &data);
            }
        }
        {
            let vertex_data = vec!(
                VertexP {
                    position: [0.0, 0.0, 0.0], 
                    color: [1.0, 0.0, 0.0, 1.0],
                }, VertexP {
                    position: [0.5, 0.0, 0.0], 
                    color: [1.0, 0.0, 0.0, 1.0],
                }, VertexP {
                    position: [- 1.0, - 1.0, - 1.0], 
                    color: [1.0, 0.0, 1.0, 1.0],
                },
            );
            let (vbuf, slice) = factory.create_vertex_buffer_with_slice(&vertex_data, &[0u32, 1u32, 2u32, 0u32][..]);
            let data = pipe_p::Data {
                vbuf: vbuf, 
                out_color: main_color.clone(),
                out_depth: main_depth.clone(),
            };
            encoder.draw(&slice, &pso_p, &data);
        }
        // {
        //     let data = pipe_w2::Data {
        //         vbuf: font_entry.vertex_buffer.clone(),
        //         u_model_view_proj: camera.projection.into(),
        //         u_model_view: camera.view.into(),
        //         u_light: [1.0, 0.5, -0.5f32],
        //         u_ambient_color: [0.00, 0.00, 0.01, 0.4],
        //         u_eye_direction: camera.direction().into(),
        //         u_texture: (font_entry.texture.clone(), sampler.clone()),
        //         out_color: main_color.clone(),
        //         out_depth: main_depth.clone()
        //     };
        //     encoder.draw(&font_entry.slice, &pso_w2, &data);
        // }
        encoder.flush(&mut device);
    
        let _ = window.swap_buffers();
        device.cleanup();
    
        timer.elapsed().as_f64();
    }
}

enum GameCommand {
    CameraMove (Vector3<f32>),
    AvatorMove (Vector3<f32>),
    Exit,
}

fn event_handler(ev :glutin::Event) -> Option<GameCommand> {
    match ev {
        glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
            glutin::Event::Closed => Some(GameCommand::Exit),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::L)) 
                => Some(GameCommand::CameraMove(Vector3::new(0.5,0.0,0.0))),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::H)) 
                => Some(GameCommand::CameraMove(Vector3::new(-0.5,0.0,0.0))),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::J)) 
                => Some(GameCommand::CameraMove(Vector3::new(0.0,0.0,-0.5))),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::K)) 
                => Some(GameCommand::CameraMove(Vector3::new(0.0,0.0,0.5))),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::W)) 
                => Some(GameCommand::AvatorMove(Vector3::new(0.0, 0.1, 0.0))),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::S)) 
                => Some(GameCommand::AvatorMove(Vector3::new(0.0, -0.1, 0.0))),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::A)) 
                => Some(GameCommand::AvatorMove(Vector3::new(-0.1, 0.0, 0.0))),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::D)) 
                => Some(GameCommand::AvatorMove(Vector3::new(0.1, 0.0, 0.0))),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::Z)) 
                => Some(GameCommand::CameraMove(Vector3::new(0.0,0.5,0.0))),
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::X)) 
                => Some(GameCommand::CameraMove(Vector3::new(0.0,-0.5,0.0))),
            _   => { None }
    }
}

struct Image<T: gfx::format::TextureFormat> {
    data: Vec<u8>,
    width: u16,
    height: u16,
    format: std::marker::PhantomData<T>
}

gfx_defines!{
    pipeline pipe_w {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        u_model_view_proj: gfx::Global<[[f32; 4]; 4]> = "u_model_view_proj",
        u_model_view: gfx::Global<[[f32; 4]; 4]> = "u_model_view",
        u_light: gfx::Global<[f32; 3]> = "u_light",
        u_ambient_color: gfx::Global<[f32; 4]> = "u_ambientColor",
        u_eye_direction: gfx::Global<[f32; 3]> = "u_eyeDirection",
        u_texture: gfx::TextureSampler<[f32; 4]> = "u_texture",
        out_color: gfx::RenderTarget<ColorFormat> = "Target0",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
        b_skinning: gfx::RawConstantBuffer = "b_skinning",
    }
    vertex Vertex {
        position: [f32; 3] = "position",
        normal: [f32; 3] = "normal",
        uv: [f32; 2] = "uv",
        joint_indices: [i32; 4] = "joint_indices",
        joint_weights: [f32; 4] = "joint_weights",
    }
    pipeline pipe_p {
        vbuf: gfx::VertexBuffer<VertexP> = (),
        out_color: gfx::RenderTarget<ColorFormat> = "Target0",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
    vertex VertexP {
        position: [f32; 3] = "position",
        color: [f32; 4] = "color",
    }
    pipeline pipe_w2 {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        u_model_view_proj: gfx::Global<[[f32; 4]; 4]> = "u_model_view_proj",
        u_model_view: gfx::Global<[[f32; 4]; 4]> = "u_model_view",
        u_light: gfx::Global<[f32; 3]> = "u_light",
        u_ambient_color: gfx::Global<[f32; 4]> = "u_ambientColor",
        u_eye_direction: gfx::Global<[f32; 3]> = "u_eyeDirection",
        u_texture: gfx::TextureSampler<f32> = "u_texture",
        out_color: gfx::RenderTarget<ColorFormat> = "Target0",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
    constant Skinning {
        transform: [[f32; 4]; 4] = "u_transform",
    }
}

struct Camera<T> {
    position: Point3<T>,
    target: Point3<T>,
    // up: Vector3<T>,
    view: Matrix4<T>,
    perspective: Matrix4<T>,
    projection: Matrix4<T>
}


impl<T: cgmath::BaseFloat> Camera<T> {
    fn new(position: Point3<T>, target: Point3<T>, perspective: cgmath::PerspectiveFov<T>) -> Camera<T> {
        let view = Matrix4::look_at(position,
                                    target,
                                    Vector3::new(Zero::zero(), Zero::zero(), One::one()));
        let perspective = Matrix4::from(perspective);
        Camera {
            position: position,
            target: target,
            view: view,
            perspective: perspective,
            projection: perspective * view
        }
    }
    fn look_to(&mut self, target: Point3<T>) {
        self.target = target;
    }
    fn direction(& self) -> Vector3<T> {
        self.target - self.position
    }
    fn update(&mut self) {
        self.view = Matrix4::look_at(self.position,
                                     self.target,
                                     Vector3::new(Zero::zero(), Zero::zero(), One::one()));
        self.projection = self.perspective * self.view;
    }
}

impl Default for Vertex {
    fn default() -> Vertex {
        Vertex {
            position: [0.0; 3],
            normal: [0.0; 3],
            uv: [0.0; 2],
            joint_indices: [0; 4],
            joint_weights: [0.0; 4]
        }
    }
}

const CLEAR_COLOR: [f32; 4] = [0.1, 0.2, 0.3, 1.0];

pub struct Entry<R: gfx::Resources, V, View> {
    slice: gfx::Slice<R>,
    vertex_buffer: gfx::handle::Buffer<R, V>,
    texture: gfx::handle::ShaderResourceView<R, View>
}

#[derive(Debug)]
struct Joint {
    pub global : Matrix4<f32>,
    bind: Matrix4<f32>,
    inverse: Matrix4<f32>
}

fn entry<R: gfx::Resources, F: gfx::Factory<R>, V, T: gfx::format::TextureFormat>(factory: &mut F, vertex_data: &[V], img: Image<T>) -> Entry<R, V, T::View> 
where V: gfx::traits::Pod + gfx::pso::buffer::Structure<gfx::format::Format> {
    let index_data: Vec<u32> = vertex_data.iter().enumerate().map(|(i, _)| i as u32).collect();
    entry_(factory, &vertex_data, &index_data[..], img)
}


fn entry_<R: gfx::Resources, F: gfx::Factory<R>, V, T: gfx::format::TextureFormat>(factory: &mut F, vertex_data: &[V], index_data: &[u32], img: Image<T>) -> Entry<R, V, T::View> 
where V: gfx::traits::Pod + gfx::pso::buffer::Structure<gfx::format::Format> {
    use gfx::traits::FactoryExt;
    let (vbuf, slice) = factory.create_vertex_buffer_with_slice(&vertex_data, index_data);

    let tex_kind = gfx::texture::Kind::D2(img.width, img.height, gfx::texture::AaMode::Single);
    let (_, view) = factory.create_texture_immutable_u8::<T>(tex_kind, &[&img.data]).expect("create texture failure");

    Entry {
        slice: slice,
        vertex_buffer: vbuf,
        texture: view
    }
}

fn font_entry<R: gfx::Resources, F: gfx::Factory<R>>(factory: &mut F, text: &str) -> Entry<R, Vertex, f32> {
    let chars: Vec<char> = "雑に文字描画abcdef12345".chars().map(|c| c).collect();
    let font_entry = match Font::from_path("assets/VL-PGothic-Regular.ttf", 8, Some(&chars[..])) {
        Ok(font) => { Some(font_entry_(factory, font, text)) },
        Err(err) => { println!("{:?}", err); None }
    }.unwrap();
    font_entry
}

fn font_entry_<R: gfx::Resources, F: gfx::Factory<R>>(factory: &mut F, font: Font, text: &str) -> Entry<R, Vertex, f32>  {
    let mut vertex_data = Vec::new();
    let mut index_data = Vec::new();

    let (mut x, z, y) = (0.0, 0.0, 0.0);
    for ch in text.chars() {
        let ch_info = match font.chars.get(&ch) {
            Some(info) => info,
            None => continue,
        };
        let x_offset = x + ch_info.x_offset as f32;
        let y_offset = y + ch_info.y_offset as f32;
        let tex = ch_info.tex;

        let index = vertex_data.len() as u32;

        vertex_data.push(
            Vertex { 
                position: [x_offset, z, y_offset],
                normal: [ 0.0, 0.0, 1.0],
                uv: [tex[0], tex[1] + ch_info.tex_height] ,
                joint_indices: [0;4], joint_weights: [0.0;4]
            }
        );
        vertex_data.push(
            Vertex { 
                position: [x_offset, z, y_offset + ch_info.height as f32],
                normal: [ 0.0, 0.0, 1.0],
                uv: [tex[0], tex[1]], 
                joint_indices: [0;4], joint_weights: [0.0;4]
            }
        );
        vertex_data.push(
            Vertex { 
                position: [x_offset + ch_info.width as f32, z, y_offset + ch_info.height as f32],
                normal: [0.0, 0.0, 1.0],
                uv: [tex[0] + ch_info.tex_width, tex[1]], 
                joint_indices: [0;4], joint_weights: [0.0;4]
            }
        );
        vertex_data.push(
            Vertex { 
                position: [x_offset + ch_info.width as f32, z, y_offset],
                normal: [0.0, 0.0, 1.0],
                uv: [tex[0] + ch_info.tex_width, tex[1] + ch_info.tex_height] ,
                joint_indices: [0;4], joint_weights: [0.0;4]
            }
        );
        index_data.push(index + 0);
        index_data.push(index + 1);
        index_data.push(index + 3);
        index_data.push(index + 3);
        index_data.push(index + 1);
        index_data.push(index + 2);

        x += ch_info.x_advance as f32;
    }
    entry_(factory,
           vertex_data.as_slice(), index_data.as_slice(),
           Image::<(gfx::format::R8, gfx::format::Unorm)>::from(font))
}

fn query_entry<R: gfx::Resources, F: gfx::Factory<R>>(conn: &Connection, factory: &mut F, ids: &[i32]) -> RusqliteResult<HashMap<i32, Object<R, Vertex>>> {
    let mut result = HashMap::default();

    for id in ids {
        let mut entries = Vec::new();
        let meshes = query_mesh(&conn, id)?;
        let joints = query_skeleton(&conn, id)?;
        for r in meshes
        {
            let vertex_data = r.0;
            let texture_id = r.1;

            let img = query_texture(&conn, texture_id).unwrap();

            entries.push(
                entry(factory, vertex_data.as_slice(), img)
                );
        }
        result.insert(id.clone(), 
                      Object {
                          entries: entries,
                          position: Point3::new(0.0, 0.0, 0.0),
                          // front: Vector3::new(0.0, -1.0, 0.0)
                          skinning: joints.iter().map(|(i, j)| {
                                                      Skinning{ transform: j.global.into()}
                          }).collect()
                      });
    }
    Ok(result)
}

struct Object<R: gfx::Resources, V> {
    entries: Vec<Entry<R, V, [f32;4]>>,
    position: Point3<f32>,
    // front: Vector3<f32>,
    skinning: Vec<Skinning>
}

trait Translate<T: cgmath::BaseFloat> {
    fn translate(&mut self, v: Vector3<T>);
}

impl<R: gfx::Resources, V> Translate<f32> for Object<R, V>{
    fn translate(&mut self, v: Vector3<f32>) {
        self.position += v;
    }
}
impl<T: cgmath::BaseFloat> Translate<T> for Camera<T> {
    fn translate(&mut self, v: Vector3<T>) {
        self.position += v;
    }
}


fn query_mesh(conn: &Connection, object_id: &i32) -> RusqliteResult<Vec<(Vec<Vertex>, i32)>> {
    let mut stmt = conn.prepare("
SELECT 
  M.MeshId
, M.TextureId
, MV.PositionX   
, MV.PositionY   
, MV.PositionZ   
, MV.NormalX     
, MV.NormalY     
, MV.NormalZ     
, MV.U           
, MV.V           
, MV.Joint1      
, MV.Joint2      
, MV.Joint3      
, MV.Joint4      
, MV.JointWeight1
, MV.JointWeight2
, MV.JointWeight3
, MV.JointWeight4
  FROM Object AS O
LEFT JOIN Mesh AS M
  ON O.ObjectId = M.ObjectId
LEFT JOIN MeshVertex AS MV
  ON M.ObjectId = MV.ObjectId
  and M.MeshId = MV.MeshId
WHERE O.ObjectId = ?1
Order By MV.ObjectId, MV.MeshId, MV.IndexNo
")?;
    let result = stmt.query_map(&[object_id], |r| {
        ( r.get::<&str,i32>("MeshId") as usize,
          r.get::<&str,i32>("TextureId"),
          Vertex { 
              position: [ r.get::<&str,f64>("PositionX") as f32,
                          r.get::<&str,f64>("PositionY") as f32,
                          r.get::<&str,f64>("PositionZ") as f32],
              normal: [ r.get::<&str,f64>("NormalX") as f32,
                        r.get::<&str,f64>("NormalY") as f32,
                        r.get::<&str,f64>("NormalZ") as f32],
              uv: [ r.get::<&str,f64>("U") as f32,
                    1.0 - r.get::<&str,f64>("V") as f32],
              joint_indices: [ r.get::<&str,i32>("Joint1"),
                               r.get::<&str,i32>("Joint2"),
                               r.get::<&str,i32>("Joint3"),
                               r.get::<&str,i32>("Joint4")],
              joint_weights: [ r.get::<&str,f64>("JointWeight1") as f32,
                               r.get::<&str,f64>("JointWeight2") as f32,
                               r.get::<&str,f64>("JointWeight3") as f32,
                               r.get::<&str,f64>("JointWeight4") as f32]
          }
        )
    })?;

    let mut meshes = Vec::new();
    for r in result
    {
        let (mesh_id, texture_id, v) = r?;
        if meshes.len() < mesh_id
        { 
            meshes.push((Vec::new(), texture_id));
        }
        (meshes[mesh_id - 1]).0.push(v);
    }
    Ok(meshes)
}

fn query_texture(conn: &Connection, texture_id: i32) -> RusqliteResult<Image<ColorFormat>> {
    conn.query_row("
SELECT 
  T.Width
, T.Height
, T.Data
FROM Texture AS T
WHERE T.TextureId = ?1
", &[&texture_id], |r| {
        Image {
            data: r.get::<&str, Vec<u8>>("Data"),
            width: r.get::<&str, i32>("Width") as u16, 
            height: r.get::<&str, i32>("Height") as u16,
            format: std::marker::PhantomData::<ColorFormat>
        }
    })
}

fn query_skeleton(conn: &Connection, object_id: &i32) -> RusqliteResult<HashMap<i32, Joint>> {
    let mut stmt = conn.prepare("
SELECT
  JointIndex,
  ParentIndex,
  BindPose11,
  BindPose12,
  BindPose13,
  BindPose14,
  BindPose21,
  BindPose22,
  BindPose23,
  BindPose24,
  BindPose31,
  BindPose32,
  BindPose33,
  BindPose34,
  BindPose41,
  BindPose42,
  BindPose43,
  BindPose44,
  InverseBindPose11,
  InverseBindPose12,
  InverseBindPose13,
  InverseBindPose14,
  InverseBindPose21,
  InverseBindPose22,
  InverseBindPose23,
  InverseBindPose24,
  InverseBindPose31,
  InverseBindPose32,
  InverseBindPose33,
  InverseBindPose34,
  InverseBindPose41,
  InverseBindPose42,
  InverseBindPose43,
  InverseBindPose44
  FROM Joint AS J
WHERE J.ObjectId = ?1
ORDER BY JointIndex
")?;
    let result = stmt.query_map(&[object_id], |r| {
        ( r.get::<&str,i32>("JointIndex"),
          r.get::<&str,i32>("ParentIndex"),
          Matrix4::new(r.get::<&str,f64>("BindPose11") as f32,
                       r.get::<&str,f64>("BindPose12") as f32,
                       r.get::<&str,f64>("BindPose13") as f32,
                       r.get::<&str,f64>("BindPose14") as f32,
                       r.get::<&str,f64>("BindPose21") as f32,
                       r.get::<&str,f64>("BindPose22") as f32,
                       r.get::<&str,f64>("BindPose23") as f32,
                       r.get::<&str,f64>("BindPose24") as f32,
                       r.get::<&str,f64>("BindPose31") as f32,
                       r.get::<&str,f64>("BindPose32") as f32,
                       r.get::<&str,f64>("BindPose33") as f32,
                       r.get::<&str,f64>("BindPose34") as f32,
                       r.get::<&str,f64>("BindPose41") as f32,
                       r.get::<&str,f64>("BindPose42") as f32,
                       r.get::<&str,f64>("BindPose43") as f32,
                       r.get::<&str,f64>("BindPose44") as f32),
          Matrix4::new(r.get::<&str,f64>("InverseBindPose11") as f32,
                       r.get::<&str,f64>("InverseBindPose12") as f32,
                       r.get::<&str,f64>("InverseBindPose13") as f32,
                       r.get::<&str,f64>("InverseBindPose14") as f32,
                       r.get::<&str,f64>("InverseBindPose21") as f32,
                       r.get::<&str,f64>("InverseBindPose22") as f32,
                       r.get::<&str,f64>("InverseBindPose23") as f32,
                       r.get::<&str,f64>("InverseBindPose24") as f32,
                       r.get::<&str,f64>("InverseBindPose31") as f32,
                       r.get::<&str,f64>("InverseBindPose32") as f32,
                       r.get::<&str,f64>("InverseBindPose33") as f32,
                       r.get::<&str,f64>("InverseBindPose34") as f32,
                       r.get::<&str,f64>("InverseBindPose41") as f32,
                       r.get::<&str,f64>("InverseBindPose42") as f32,
                       r.get::<&str,f64>("InverseBindPose43") as f32,
                       r.get::<&str,f64>("InverseBindPose44") as f32)
        )
    })?;

    let mut joints = HashMap::<i32, Joint>::default();
    let local: Matrix4<f32> = cgmath::One::one();
    for r in result
    {
        let (inx, p, bind, inverse) = r?;

        local.concat(&(bind * inverse));

        let m = if p != 255 {
            joints.get(&p).unwrap().global
        } else { 
            cgmath::One::one()
            // Matrix4::new(
            //   1.0, 0.0, 0.0, 0.0,
            //   0.0, (std::f32::consts::PI / 2.0).cos(), (std::f32::consts::PI / 2.0).sin(), 0.0,
            //   0.0, (-std::f32::consts::PI / 2.0).sin(), (std::f32::consts::PI / 2.0).cos(), 0.0,
            //   0.0, 0.0, 0.0, 1.0
            // )
        } * local;

        joints.insert(inx,
                      Joint {
                          global: m,
                          bind: bind,
                          inverse : inverse,
                      });
    }
    Ok(joints)
}



pub struct Font {
    width: u16,
    height: u16,
    image: Vec<u8>,
    chars: HashMap<char, BitmapChar>
}

struct BitmapChar {
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

        let mut chars_info = HashMap::default();
        let mut sum_image_width = 0;
        let mut max_ch_width = 0;
        let mut max_ch_height = 0;
        for ch in needed_chars {
            try!(face.load_char(ch as usize, ft::face::RENDER));

            let glyph = face.glyph();
            let bitmap = glyph.bitmap();

            let ch_width = bitmap.width();
            let ch_height = bitmap.rows();

            chars_info.insert(ch, BitmapChar {
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

        for (_, ch_info) in chars_info.iter_mut() {
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

        for (_, ch_info) in chars_info.iter_mut() {
            ch_info.tex[0] /= image_width as f32;
            ch_info.tex[1] /= image_height as f32;
            ch_info.tex_width = ch_info.width as f32 / image_width as f32;
            ch_info.tex_height = ch_info.height as f32 / image_height as f32;
        }

        Ok(Font{
            width: image_width as u16,
            height: image_height as u16,
            image: image,
            chars: chars_info
        })
    }
    fn get_all_face_chars<'a>(face: &mut Face<'a>) -> HashSet<char> {
        use std::char::from_u32;
        let mut result = HashSet::default();
        let mut index = 0;
        let mut face_ptr = face.raw_mut();
        unsafe {
            let mut code = ft::ffi::FT_Get_First_Char(face_ptr, &mut index);
            while index != 0 {
                from_u32(code as u32).map(|ch| result.insert(ch));
                code = ft::ffi::FT_Get_Next_Char(face_ptr, code, &mut index);
            }
        }
        result
    }
}

