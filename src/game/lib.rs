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

use gfx::{Adapter, CommandQueue, Device, FrameSync, GraphicsPoolExt,
          Surface, Swapchain, WindowExt};
use gfx::memory::Typed;
use gfx::format::Formatted;

pub type ColorFormat = gfx::format::Srgba8;
pub type DepthFormat = gfx::format::DepthStencil;

// TODO push down automaton

use cgmath:: {
    EuclideanSpace,
    Point3,
    Vector3,
    Matrix4,
    One,
    Zero,
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

pub struct App<R: gfx::Resources, B: gfx::Backend> {
    pso: gfx::PipelineState<R, pipe_w::Meta>,
    pso_w2: gfx::PipelineState<R, pipe_w2::Meta>,
    pso_p: gfx::PipelineState<R, pipe_p::Meta>,
    sampler: gfx::handle::Sampler<R>,
    invoker: CommandInterpreter<R, Vertex>,
    views: Vec<(gfx::handle::RenderTargetView<R, ColorFormat>,
                gfx::handle::DepthStencilView<R, DepthFormat>)>,
    skinning_buffer: gfx::handle::Buffer<R, Skinning>,
    device: gfx_device_gl::Device,
    graphics_pool: gfx::GraphicsCommandPool<B>,

    swap_chain: gfx_window_glutin::Swapchain,

    frame_semaphore: gfx::handle::Semaphore<R>,
    draw_semaphore: gfx::handle::Semaphore<R>,

    frame_fence: gfx::handle::Fence<R>,
    graphics_queue: gfx::queue::GraphicsQueue<B>,
}

impl App<gfx_device_gl::Resources, gfx_device_gl::Backend> {

    pub fn new(
        window: glutin::GlWindow,
        width: u32,
        height: u32,
    ) -> App<gfx_device_gl::Resources, gfx_device_gl::Backend> {
        use gfx::traits::DeviceExt;
        use gfx::Device;

        let (mut surface, adapters) = gfx_window_glutin::Window::new(window).get_surface_and_adapters();
        let gfx::Gpu { mut device, mut graphics_queues, .. } = 
            adapters[0].open_with(|family, ty| {
                (
                    (ty.supports_graphics() && surface.supports_queue(&family)) as u32,
                    gfx::QueueType::Graphics
                )
            });
        let graphics_queue = graphics_queues.pop().expect("Unable to find a graphics queue.");

        let config = gfx::SwapchainConfig::new()
            .with_color::<ColorFormat>()
            .with_depth_stencil::<DepthFormat>();
        let mut swap_chain = surface.build_swapchain(config, &graphics_queue);

        let views: Vec<(gfx::handle::RenderTargetView<_, ColorFormat>,
                        gfx::handle::DepthStencilView<_, DepthFormat>)> = swap_chain
            .get_backbuffers()
            .iter()
            .map(|&(ref color, ref ds)| {
                let color_desc = gfx::texture::RenderDesc {
                    channel: ColorFormat::get_format().1,
                    level: 0,
                    layer: None,
                };
                let rtv = device.view_texture_as_render_target_raw(color, color_desc).expect("rtv");
                let ds_desc = gfx::texture::DepthStencilDesc {
                    level: 0,
                    layer: None,
                    flags: gfx::texture::DepthStencilFlags::empty(),
                };
                let dsv = device.view_texture_as_depth_stencil_raw(
                    ds.as_ref().expect("ds"),
                    ds_desc
                ).expect("dsv");

                (Typed::new(rtv), Typed::new(dsv))
            }).collect();


        let pso = {
            let shaders = device.create_shader_set(
          b"#version 150 core
            
            uniform mat4 u_model_view_proj;
            uniform mat4 u_model_view;
            uniform b_skinning {
                mat4 u_skinning[64];
            };
            
            in vec3 position, normal;
            in vec2 uv;
            in ivec4 joint_indices;
            in vec4 joint_weights;
            
            
            out vec2 v_TexCoord;
            out vec3 _normal;
            
            void main() {
            
                vec4 bindVertex = vec4(position, 1.0);
                vec4 bindNormal = vec4(normal, 0.0);
                vec4 v =  joint_weights.x * u_skinning[joint_indices.x] * bindVertex;
                     v += joint_weights.y * u_skinning[joint_indices.y] * bindVertex;
                     v += joint_weights.z * u_skinning[joint_indices.z] * bindVertex;
                     v += joint_weights.a * u_skinning[joint_indices.a] * bindVertex;
                vec4 n = bindNormal * u_skinning[joint_indices.x] * joint_weights.x;
                n += bindNormal * u_skinning[joint_indices.y] * joint_weights.y;
                n += bindNormal * u_skinning[joint_indices.z] * joint_weights.z;
                n += bindNormal * u_skinning[joint_indices.a] * joint_weights.a;
            
                gl_Position = u_model_view_proj * v;
                v_TexCoord = uv;
                _normal = normalize(bindNormal).xyz;
            }
            ",
          b"#version 150 core
            
            uniform vec3 u_light;
            uniform vec4 u_ambientColor;
            uniform vec3 u_eyeDirection;
            uniform sampler2D u_texture;
            
            in vec2 v_TexCoord;
            in vec3 _normal;
            out vec4 Target0;
            
            void main() {
                vec4 texColor = texture(u_texture, v_TexCoord);
            
                float diffuse = clamp(dot(_normal, -u_light), 0.05f, 1.0f);
                vec3 halfLE = normalize(u_eyeDirection);
                float specular = pow(clamp(dot(_normal, halfLE), 0.0, 1.0), 50.0);
                Target0 = texColor * vec4(vec3(diffuse), 1.0) + vec4(vec3(specular), 1.0) + u_ambientColor;
            }
            ").expect("shader exists?");
            device.create_pipeline_state(
                &shaders,
                gfx::Primitive::TriangleList,
                gfx::state::Rasterizer::new_fill(),
                pipe_w::new()
                ).expect("failed to create pipeline w")
        };
        let pso_w2 = {
            let shaders = device.create_shader_set(b"
            #version 150 core
            
            uniform mat4 u_model_view_proj;
            uniform mat4 u_model_view;
            
            in vec3 position, normal;
            in vec2 uv;
            
            out vec2 v_TexCoord;
            out vec3 _normal;
            
            void main() {
                v_TexCoord = vec2(uv.x, uv.y);
            
                gl_Position = u_model_view_proj * vec4(position, 1.0);
                _normal = normalize(normal);
            }
            ",
            b"
            #version 150 core
            
            uniform vec3 u_light;
            uniform vec4 u_ambientColor;
            uniform vec3 u_eyeDirection;
            uniform sampler2D u_texture;
            
            in vec2 v_TexCoord;
            in vec3 _normal;
            out vec4 Target0;
            
            void main() {
                vec4 texColor = texture(u_texture, v_TexCoord);
            
                float diffuse = clamp(dot(_normal, -u_light), 0.05f, 1.0f);
                vec3 halfLE = normalize(u_eyeDirection);
                float specular = pow(clamp(dot(_normal, halfLE), 0.0, 1.0), 50.0);
                Target0 = texColor * vec4(vec3(diffuse), 1.0) + vec4(vec3(specular), 1.0) + u_ambientColor;
            }
            ",
            ).expect("shader exists?");
            device.create_pipeline_state(
                &shaders,
                gfx::Primitive::TriangleList,
                gfx::state::Rasterizer::new_fill(),
                pipe_w2::new()
                ).expect("failed to create pipeline w2")
        };
        let pso_p = {
            let shaders = device.create_shader_set(b"
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
            device.create_pipeline_state(
                &shaders,
                gfx::Primitive::LineStrip,
                gfx::state::Rasterizer::new_fill().with_cull_back(),
                pipe_p::new()
                ).expect("failed to create pipeline p")
        };

        let graphics_pool = graphics_queue.create_graphics_pool(1);
            
        let sampler = {
            let sampler_info = gfx::texture::SamplerInfo::new(
                gfx::texture::FilterMethod::Trilinear,
                gfx::texture::WrapMode::Clamp);
            device.create_sampler(sampler_info)
        };
        
        let skinning_buffer = device.create_constant_buffer::<Skinning>(64);
        let invoker = CommandInterpreter::new(
            &mut device,
            (width as f32) / (height as f32)
        );

        let frame_semaphore = device.create_semaphore();
        let draw_semaphore = device.create_semaphore();
        let frame_fence = device.create_fence(false);

        App {
            device,
            sampler,
            pso,
            pso_w2,
            pso_p,
            skinning_buffer,
            invoker,
            frame_semaphore,
            draw_semaphore,
            frame_fence,
            graphics_pool,
            swap_chain,
            graphics_queue,
            views,
        }
    }

    pub fn handle_input(&mut self, ev :glutin::WindowEvent) {
        self.invoker.handle_input(ev)
    }

    fn pre_render(&mut self) {
        self.invoker.execute_all_commands()
    }

    pub fn render(&mut self) {
        self.pre_render();

        let elapsed = self.invoker.system.target.timer.elapsed().as_f64();

        let sampler = &mut self.sampler;

        let frame = self.swap_chain.acquire_frame(FrameSync::Semaphore(&self.frame_semaphore));
        let view = self.views[frame.id()].clone();
        {
            let mut encoder = self.graphics_pool.acquire_graphics_encoder();

            encoder.clear(&view.0.clone(), CLEAR_COLOR);
            encoder.clear_depth(&view.1.clone(), 1.0);


            for obj in self.invoker.target().values() {
                let camera = self.invoker.camera(); 
                let mv = camera.view * Matrix4::from_translation(obj.position.to_vec());
                let mvp = camera.perspective * mv;
                {
                    let a = obj.get_skinning(elapsed);
                    encoder.update_buffer(&self.skinning_buffer, &a[..], 0).expect("ub");
                }
                for entry in &obj.entries {
                    let data = pipe_w::Data {
                        vbuf: entry.vertex_buffer.clone(),
                        u_model_view_proj: mvp.into(),
                        u_model_view: mv.into(),
                        u_light: [0.2, 0.2, -0.2f32],
                        u_ambient_color: [0.01, 0.01, 0.01, 1.0],
                        u_eye_direction: camera.direction().into(),
                        u_texture: (entry.texture.clone(), sampler.clone()),
                        out_color: view.0.clone(),
                        out_depth: view.1.clone(),
                        b_skinning: self.skinning_buffer.raw().clone(),
                    };
                    encoder.draw(&entry.slice, &self.pso, &data);
                }
            }
            {
                let camera = self.invoker.camera(); 
                let font_entry = font_entry(&mut self.device, &format!("{:?}", elapsed)).expect("fe");

                let data = pipe_w2::Data {
                    vbuf: font_entry.vertex_buffer.clone(),
                    u_model_view_proj: camera.projection.into(),
                    u_model_view: camera.view.into(),
                    u_light: [1.0, 0.5, -0.5f32],
                    u_ambient_color: [0.00, 0.00, 0.01, 0.4],
                    u_eye_direction: camera.direction().into(),
                    u_texture: (font_entry.texture.clone(), sampler.clone()),
                    out_color: view.0.clone(),
                    out_depth: view.1.clone()
                };
                encoder.draw(&font_entry.slice, &self.pso_w2, &data);
            }
            encoder.synced_flush(&mut self.graphics_queue, &[&self.frame_semaphore], &[&self.draw_semaphore], Some(&self.frame_fence))
                .expect("Colud not flush encoder");
        }
        self.swap_chain.present(&mut self.graphics_queue, &[&self.draw_semaphore]);
        self.device.wait_for_fences(&[&self.frame_fence], gfx::WaitFor::All, 1_000_000);
        self.graphics_queue.cleanup();
        self.graphics_pool.reset();
    }
}


enum AvatorCommand {
    Move (Vector3<f32>),
}
enum CameraCommand {
    Move (Vector3<f32>),
    // LookTo (Vector3<f32>),
}
enum SystemCommand {
    Exit
}

trait Command<T> {
    fn get_level(&self) -> Level;
    fn execute(&self, &mut T);
}


struct Invoker<Cmd, T> {
    commands: Vec<Cmd>,
    target: T,
    current_index: usize,
}

struct System {
    timer: coarsetime::Instant,
}

struct CommandInterpreter<R: gfx::Resources, V> {
    camera: Invoker<CameraCommand, Camera<f32>>,
    avators: Invoker<AvatorCommand, HashMap<i32, Object<R, V>>>,
    system: Invoker<SystemCommand, System>,
}

impl<R: gfx::Resources> CommandInterpreter<R, Vertex> {
    fn new<F: gfx::Device<R>>(device: &mut F, aspect: f32) -> Self {
        let conn = Connection::open(&Path::new("file.db")).expect("failed to open sqlite file");
        CommandInterpreter {
            avators: Invoker::<AvatorCommand, HashMap<i32, Object<R, _>>>::new(
                        query_entry::<R, F>(&conn, device, &[1,2]).unwrap()),
            camera: Invoker::<CameraCommand, Camera<f32>>::new(
                        Camera::new(
                            Point3::new(30.0, -40.0, 30.0),
                            Point3::new(0.0, 0.0, 0.0),
                            cgmath::PerspectiveFov {
                                fovy: cgmath::Rad(16.0f32.to_radians()),
                                aspect: aspect,  
                                near: 5.0,
                                far: 1000.0,
                        })
                ) , 
            system: Invoker::<SystemCommand, System>::new(System {
                timer: coarsetime::Instant::now()
            })
        }
    }
    fn target(&self) -> &HashMap<i32, Object<R, Vertex>> {
        &self.avators.target
    }
    fn camera(&self) -> &Camera<f32> {
        &self.camera.target
    }
    fn handle_input(&mut self, ev: glutin::WindowEvent) {
        match ev {
                glutin::WindowEvent::KeyboardInput {
                    input: glutin::KeyboardInput {
                        state: glutin::ElementState::Pressed,
                        virtual_keycode: Some(glutin::VirtualKeyCode::L), ..
                    }, ..
                } => self.avators.append_command(AvatorCommand::Move(Vector3::new(0.5,0.0,0.0))),
                glutin::WindowEvent::KeyboardInput {
                    input: glutin::KeyboardInput {
                        state: glutin::ElementState::Pressed,
                        virtual_keycode: Some(glutin::VirtualKeyCode::H), ..
                    }, ..
                } => self.avators.append_command(AvatorCommand::Move(Vector3::new(-0.5,0.0,0.0))),
                glutin::WindowEvent::KeyboardInput {
                    input: glutin::KeyboardInput {
                        state: glutin::ElementState::Pressed,
                        virtual_keycode: Some(glutin::VirtualKeyCode::J), ..
                    }, ..
                } => self.avators.append_command(AvatorCommand::Move(Vector3::new(0.0,-0.5,0.0))),
                glutin::WindowEvent::KeyboardInput {
                    input: glutin::KeyboardInput {
                        state: glutin::ElementState::Pressed,
                        virtual_keycode: Some(glutin::VirtualKeyCode::K), ..
                    }, ..
                } => self.avators.append_command(AvatorCommand::Move(Vector3::new(0.0,0.5,0.0))),
                glutin::WindowEvent::KeyboardInput {
                    input: glutin::KeyboardInput {
                        state: glutin::ElementState::Pressed,
                        virtual_keycode: Some(glutin::VirtualKeyCode::W), ..
                    }, ..
                } => self.camera.append_command(CameraCommand::Move(Vector3::new(0.0, 0.1, 0.0))),
                glutin::WindowEvent::KeyboardInput {
                    input: glutin::KeyboardInput {
                        state: glutin::ElementState::Pressed,
                        virtual_keycode: Some(glutin::VirtualKeyCode::S), ..
                    }, ..
                } => self.camera.append_command(CameraCommand::Move(Vector3::new(0.0, -0.1, 0.0))),
                glutin::WindowEvent::KeyboardInput {
                    input: glutin::KeyboardInput {
                        state: glutin::ElementState::Pressed,
                        virtual_keycode: Some(glutin::VirtualKeyCode::A), ..
                    }, ..
                } => self.camera.append_command(CameraCommand::Move(Vector3::new(-0.1, 0.0, 0.0))),
                glutin::WindowEvent::KeyboardInput {
                    input: glutin::KeyboardInput {
                        state: glutin::ElementState::Pressed,
                        virtual_keycode: Some(glutin::VirtualKeyCode::D), ..
                    }, ..
                } => self.camera.append_command(CameraCommand::Move(Vector3::new(0.1, 0.0, 0.0))),
                // glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::Z)) 
                //     => Some(GameCommand::CameraMove(Vector3::new(0.0,0.5,0.0))),
                // glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::X)) 
                //     => Some(GameCommand::CameraMove(Vector3::new(0.0,-0.5,0.0))),
                _   => { }
        }
    }

    fn execute_all_commands(&mut self) {
        self.avators.execute_all_commands();
        self.camera.execute_all_commands();
    }
}

impl<Cmd, T> Invoker<Cmd, T> {
    fn new(t: T) -> Self {
        Invoker {
            commands: Vec::new(),
            target: t,
            current_index: 0,
        }
    }
}

impl<Cmd, T> Invoker<Cmd, T>
    where Cmd: Command<T> {
    fn execute_command(&mut self) {
        if self.commands.len() <= self.current_index {
            return;
        }
        let c = &self.commands[self.current_index];
        let t = &mut self.target;

        c.execute(t);

        self.current_index += 1;
    }
    fn execute_all_commands(&mut self) {
        for _ in self.current_index..self.commands.len() {
            self.execute_command();
        }
        self.commands.clear();
        self.current_index = 0;
    }
    fn append_command(&mut self, c: Cmd) {
        self.commands.push(c);
    }
}


impl Command<Camera<f32>> for CameraCommand {
    fn get_level(&self) -> Level {
        Level::System
    }
    fn execute(&self, c: &mut Camera<f32>) {
        match *self {
            CameraCommand::Move(v) => {
                c.translate(v); 
                c.update();
            },
        }
    }
}

impl<R: gfx::Resources, V> Command<Object<R, V>> for AvatorCommand {
    fn get_level(&self) -> Level {
        Level::Avator
    }
    fn execute(&self, c: &mut Object<R, V>) {
        match *self {
            AvatorCommand::Move(v) => {
                c.translate(v); 
            },
        }
    }
}
impl<R: gfx::Resources, V> Command<HashMap<i32, Object<R, V>>> for AvatorCommand {
    fn get_level(&self) -> Level {
        Level::Avator
    }
    fn execute(&self, c: &mut HashMap<i32, Object<R, V>>) {
        match *self {
            AvatorCommand::Move(v) => {
                c.get_mut(&1).unwrap().translate(v); 
            },
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Level {
    Avator,
    World,
    System,
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

#[derive(Debug, Copy, Clone)]
struct Joint {
    joint_index: i32,
    pub global : Matrix4<f32>,
    bind: Matrix4<f32>,
    parent: i32,
    inverse: Matrix4<f32>
}

#[derive(Debug)]
struct Animation {
    joint_index: i32,
    time: f32,
    pose: Matrix4<f32>,
}

fn entry<R: gfx::Resources, F: gfx::Device<R>, V, T: gfx::format::TextureFormat>(device: &mut F, vertex_data: &[V], img: Image<T>) -> Entry<R, V, T::View> 
where V: gfx::traits::Pod + gfx::pso::buffer::Structure<gfx::format::Format> {
    let index_data: Vec<u32> = vertex_data.iter().enumerate().map(|(i, _)| i as u32).collect();
    entry_(device, &vertex_data, &index_data[..], img)
}


fn entry_<R: gfx::Resources, F: gfx::Device<R>, V, T: gfx::format::TextureFormat>(device: &mut F, vertex_data: &[V], index_data: &[u32], img: Image<T>) -> Entry<R, V, T::View> 
where V: gfx::traits::Pod + gfx::pso::buffer::Structure<gfx::format::Format> {
    use gfx::traits::DeviceExt;
    let (vbuf, slice) = device.create_vertex_buffer_with_slice(&vertex_data, index_data);

    let tex_kind = gfx::texture::Kind::D2(img.width, img.height, gfx::texture::AaMode::Single);
    let (_, view) = device.create_texture_immutable_u8::<T>(tex_kind, &[&img.data]).expect("create texture failure");

    Entry {
        slice: slice,
        vertex_buffer: vbuf,
        texture: view
    }
}

fn font_entry<R: gfx::Resources, F: gfx::Device<R>>(device: &mut F, text: &str) -> Result<Entry<R, Vertex, f32>,FontError> {
    let chars: Vec<char> = "雑に文字描画abcdef0123456789.+-_".chars().map(|c| c).collect();
    Font::from_path("assets/VL-PGothic-Regular.ttf", 8, Some(&chars[..]))
    .and_then(|font| Ok(font_entry_(device, font, text)))
}

fn font_entry_<R: gfx::Resources, F: gfx::Device<R>>(device: &mut F, font: Font, text: &str) -> Entry<R, Vertex, f32>  {
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
    entry_(device,
           vertex_data.as_slice(), index_data.as_slice(),
           Image::<(gfx::format::R8, gfx::format::Unorm)>::from(font))
}

fn query_entry<R: gfx::Resources, F: gfx::Device<R>>(conn: &Connection, device: &mut F, ids: &[i32]) -> RusqliteResult<HashMap<i32, Object<R, Vertex>>> {
    let mut result = HashMap::default();

    for id in ids {
        let mut entries = Vec::new();
        let meshes = query_mesh(&conn, id)?;
        let joints = query_skeleton(&conn, id)?;
        let animations = query_animation(&conn, id)?;
        for r in meshes
        {
            let vertex_data = r.0;
            let texture_id = r.1;

            let img = query_texture(&conn, texture_id).unwrap();

            entries.push(
                entry(device, vertex_data.as_slice(), img)
                );
        }
        result.insert(id.clone(), 
                      Object {
                          entries: entries,
                          position: Point3::new(0.0, 0.0, 0.0),
                          // front: Vector3::new(0.0, -1.0, 0.0)
                          joints: joints,
                          animations: animations
                      });
    }
    Ok(result)
}

struct Object<R: gfx::Resources, V> {
    entries: Vec<Entry<R, V, [f32;4]>>,
    position: Point3<f32>,
    // front: Vector3<f32>,
    joints: Vec<Joint>,
    animations: Vec<Vec<(f32, Animation)>>,
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

impl<R: gfx::Resources, V> Object<R, V> {
    fn get_skinning(&self, time: f64) -> Vec<Skinning> {
        if self.joints.len() > 0 {
            let mut local = Vec::<Matrix4<f32>>::with_capacity(255);
            self.joints.iter().map(|j| {

                let p = if j.parent == 255 {
                    cgmath::One::one()
                } else { 
                    *local.get(j.parent as usize).unwrap()
                };
           
                match self.animations.get(j.joint_index as usize) {
                    Some(v) => {
                        let length = v.len();

                        let output = p * if length > 0 {
                            let duration = 4.0;
                            let sample_per_second = length as f32 / duration; 
                            let t = (time as f32 % duration) * sample_per_second;

                            let index_1 = t.floor() as usize;
                            let ceiled = t.ceil() as usize;
                            let index_2 = if ceiled == length { 0 } else { ceiled };

                            let blend_factor = t - index_1 as f32;

                            let sample_1 = &v[index_1];
                            let sample_2 = &v[index_2];

                            let pose_1: Matrix4<f32> = sample_1.1.pose;
                            let pose_2: Matrix4<f32> = sample_2.1.pose;

                            let pose = pose_1 + (pose_2 - pose_1) * blend_factor;

                            local.insert(j.joint_index as usize, p * pose);
                            pose * j.inverse
                        } else {
                            local.insert(j.joint_index as usize, j.bind);
                            j.bind 
                        };

                        Skinning{ 
                            transform: output.into()
                        }
                    },
                    _ => {
                        let output = j.bind;
                        local.insert(j.joint_index as usize, output);

                        Skinning{ 
                            transform: (output).into()
                        }
                    }
                }
            }).collect()
        } else { 
            let identity: Matrix4<f32> = cgmath::One::one();
            vec!({Skinning{ transform: identity.into()}})
        }
    }
    fn get_skinning_at(&self, index: usize) -> Vec<Skinning> {
        if self.joints.len() > 0 {
            let mut local = Vec::<Matrix4<f32>>::with_capacity(255);
            self.joints.iter().map(|j| {

                let p = if j.parent == 255 {
                    cgmath::One::one()
                } else { 
                    *local.get(j.parent as usize).unwrap()
                };
           
                match self.animations.get(j.joint_index as usize) {
                    Some(v) => {
                        let length = v.len();

                        let output = p * if length > 0 {
                            let inx = index % length;

                            let sample_1 = &v[inx];

                            let pose_1: Matrix4<f32> = sample_1.1.pose;
                            // println!("{}: {} :{}", inx,  sample_1.0);

                            let pose = pose_1; 

                            local.insert(j.joint_index as usize, p * pose);
                            pose * j.inverse
                        } else {
                            local.insert(j.joint_index as usize, j.bind);
                            j.bind 
                        };

                        Skinning{ 
                            transform: output.into()
                        }
                    },
                    _ => {
                        let output = j.bind;
                        local.insert(j.joint_index as usize, output);

                        Skinning{ 
                            transform: (output).into()
                        }
                    }
                }
            }).collect()
        } else { 
            let identity: Matrix4<f32> = cgmath::One::one();
            vec!({Skinning{ transform: identity.into()}})
        }
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

fn query_skeleton(conn: &Connection, object_id: &i32) -> RusqliteResult<Vec<Joint>> {
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


    let mut joints = Vec::<Joint>::with_capacity(255);
    for r in result
    {
        let (inx, p, bind, inverse) = r?;

        let joint = Joint {
            joint_index: inx,
            global: bind,
            bind: bind,
            parent: p,
            inverse : inverse,
        };

        joints.insert(inx as usize, joint);
    }
    Ok(joints)
}

fn query_animation(conn: &Connection, object_id: &i32) -> RusqliteResult<Vec<Vec<(f32, Animation)>>> {
    let mut stmt = conn.prepare("
SELECT
    AnimationId ,
    ObjectId    ,
    JointIndex  ,
    SampleTime  ,
    SamplePose11,
    SamplePose12,
    SamplePose13,
    SamplePose14,
    SamplePose21,
    SamplePose22,
    SamplePose23,
    SamplePose24,
    SamplePose31,
    SamplePose32,
    SamplePose33,
    SamplePose34,
    SamplePose41,
    SamplePose42,
    SamplePose43,
    SamplePose44,
    Name        
  FROM Animation AS A
WHERE A.ObjectId = ?1
Order By JointIndex
")?;
    let result = stmt.query_map(&[object_id], |r| {
        ( r.get::<&str,i32>("AnimationId"),
          r.get::<&str,i32>("JointIndex"),
          r.get::<&str,f64>("SampleTime"),
          Matrix4::new(r.get::<&str,f64>("SamplePose11") as f32,
                       r.get::<&str,f64>("SamplePose12") as f32,
                       r.get::<&str,f64>("SamplePose13") as f32,
                       r.get::<&str,f64>("SamplePose14") as f32,
                       r.get::<&str,f64>("SamplePose21") as f32,
                       r.get::<&str,f64>("SamplePose22") as f32,
                       r.get::<&str,f64>("SamplePose23") as f32,
                       r.get::<&str,f64>("SamplePose24") as f32,
                       r.get::<&str,f64>("SamplePose31") as f32,
                       r.get::<&str,f64>("SamplePose32") as f32,
                       r.get::<&str,f64>("SamplePose33") as f32,
                       r.get::<&str,f64>("SamplePose34") as f32,
                       r.get::<&str,f64>("SamplePose41") as f32,
                       r.get::<&str,f64>("SamplePose42") as f32,
                       r.get::<&str,f64>("SamplePose43") as f32,
                       r.get::<&str,f64>("SamplePose44") as f32)
        )
    })?;

    let mut animations = Vec::<Vec<(f32, Animation)>>::with_capacity(255);
    for r in result
    {
        let (_, joint_index, time, pose) = r?;

        if joint_index >= 0 {
            (|t: (f32, Animation) | {
                if match animations.get(joint_index as usize) { Some(_) => true, _ => false } {
                    animations.get_mut(joint_index as usize).unwrap().push(t);
                } else {
                    for i in animations.len() .. joint_index as usize {
                        animations.insert(i, vec!()); 
                    }
                    animations.insert(joint_index as usize, vec!(t)); 
                }
            })((time as f32,
                Animation {
                    joint_index: joint_index,
                    time: time as f32,
                    pose : pose,
                })
              );
        }
    }
    Ok(animations)
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
}

