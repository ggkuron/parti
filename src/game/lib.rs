#![feature(conservative_impl_trait)]
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
extern crate gilrs;

mod models;
mod font;

use rusqlite::Connection;
use rusqlite::Error as RusqliteError;
use std::path::Path;
use fnv::FnvHashMap as HashMap;

use models::*;
use font::*;

use gfx::{
    Adapter,
    CommandQueue,
    Device,
    FrameSync,
    GraphicsPoolExt,
    Surface,
    Swapchain,
    WindowExt
};
use gfx::memory::Typed;
use gfx::format::Formatted;

pub type ColorFormat = gfx::format::Srgba8;
pub type DepthFormat = gfx::format::DepthStencil;
type TextureFormat = ColorFormat;

use cgmath::{
    EuclideanSpace,
    Point3,
    Vector3,
    Matrix4,
    One,
    Zero,
};

use gilrs::{
    Gilrs,
    Button,
    Axis,
    Event
};

#[derive(Debug)]
pub enum AppError {
    RusqliteError(RusqliteError),
    FontError(FontError)
}

impl From<RusqliteError> for AppError {
    fn from(e: RusqliteError) -> AppError { AppError::RusqliteError(e) }
}
impl From<FontError> for AppError {
    fn from(e: FontError) -> AppError { AppError::FontError(e) }
}


type View<R> = (
    gfx::handle::RenderTargetView<R, ColorFormat>,
    gfx::handle::DepthStencilView<R, DepthFormat>
);

pub struct App<R: gfx::Resources, B: gfx::Backend> {
    pub running: bool,

    world: Invoker<SystemCommand, World<B, Vertex>>,
    views: Vec<View<R>>,
    device: gfx_device_gl::Device,
    graphics_pool: gfx::GraphicsCommandPool<B>,

    swap_chain: gfx_window_glutin::Swapchain,

    frame_semaphore: gfx::handle::Semaphore<R>,
    draw_semaphore: gfx::handle::Semaphore<R>,

    frame_fence: gfx::handle::Fence<R>,
    graphics_queue: gfx::queue::GraphicsQueue<B>,

    gilrs: Gilrs,
    input: GamePad,
}

impl App<gfx_device_gl::Resources, gfx_device_gl::Backend> {
    pub fn new (
        window: glutin::GlWindow,
        width: u32,
        height: u32,
    ) -> App<gfx_device_gl::Resources, gfx_device_gl::Backend> {
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

        let views: Vec<_> = swap_chain
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

        let graphics_pool = graphics_queue.create_graphics_pool(1);
            
        let world = Invoker::<SystemCommand, _>::new(
            World::new(
                &mut device,
                (width as f32) / (height as f32),
            )
        );

        let frame_semaphore = device.create_semaphore();
        let draw_semaphore = device.create_semaphore();
        let frame_fence = device.create_fence(false);

        let gilrs = Gilrs::new();

        App {
            running: true,
            device,
            world,
            frame_semaphore,
            draw_semaphore,
            frame_fence,
            graphics_pool,
            swap_chain,
            graphics_queue,
            views,
            gilrs,
            input: GamePad::default(),
        }
    }

    pub fn handle_input(&mut self, ev :glutin::WindowEvent) {
        match ev {
            glutin::WindowEvent::Closed | 
            glutin::WindowEvent::KeyboardInput {
                input: glutin::KeyboardInput {
                    state: glutin::ElementState::Pressed,
                    virtual_keycode: Some(glutin::VirtualKeyCode::Escape), ..
                }, ..
            } => self.running = false,
            glutin::WindowEvent::KeyboardInput {
                input , .. 
            } if glutin::ElementState::Pressed == input.state => {
                match input.virtual_keycode {
                    Some(glutin::VirtualKeyCode::L) => {
                         self.input.dpad_right = true 
                    },
                    Some(glutin::VirtualKeyCode::H) => {
                         self.input.dpad_left = true 
                    },
                    Some(glutin::VirtualKeyCode::J) => {
                         self.input.dpad_down = true 
                    },
                    Some(glutin::VirtualKeyCode::K) => {
                         self.input.dpad_up = true 
                    },
                    Some(glutin::VirtualKeyCode::W) => {
                         self.input.left_y = -1.0
                    },
                    Some(glutin::VirtualKeyCode::S) => {
                         self.input.left_y = -1.0
                    },
                    Some(glutin::VirtualKeyCode::A) => {
                         self.input.left_x = -1.0
                    },
                    Some(glutin::VirtualKeyCode::D) => {
                         self.input.left_x = 1.0
                    },
                    Some(glutin::VirtualKeyCode::M) => {
                         self.input.start = true
                    },
                    _ => {}
                }
            }, 
            glutin::WindowEvent::KeyboardInput {
                input , .. 
            } => {
                match input.virtual_keycode {
                    Some(glutin::VirtualKeyCode::L) => {
                         self.input.dpad_right = false 
                    },
                    Some(glutin::VirtualKeyCode::H) => {
                         self.input.dpad_left = false 
                    },
                    Some(glutin::VirtualKeyCode::J) => {
                         self.input.dpad_down = false
                    },
                    Some(glutin::VirtualKeyCode::K) => {
                         self.input.dpad_up = false 
                    },
                    Some(glutin::VirtualKeyCode::W) => {
                         self.input.left_y = 0.0
                    },
                    Some(glutin::VirtualKeyCode::S) => {
                         self.input.left_y = 0.0
                    },
                    Some(glutin::VirtualKeyCode::A) => {
                         self.input.left_x = 0.0
                    },
                    Some(glutin::VirtualKeyCode::D) => {
                         self.input.left_x = 0.0
                    },
                    Some(glutin::VirtualKeyCode::M) => {
                         self.input.start = false
                    },
                    _ => {}
                }
            },
            glutin::WindowEvent::AxisMotion {
                axis,
                value,
                ..
            } => {
                println!("axis motion {}: {}", axis, value);
            },
            _   => { }
        }
    }

    fn update(&mut self) {
        while let Some(ev) = self.gilrs.next_event() {
            self.gilrs.update(&ev);
            let id = ev.id;
            match ev.event {
                gilrs::EventType::Connected => {
                    println!("connected {}", id);
                },
                gilrs::EventType::Disconnected => {
                    println!("disconnected {}", id);
                },
                _ => {}
            }
        }

        {
            let pad = &self.gilrs[0];
            if pad.is_connected() {
                self.input = GamePad {
                    dpad_up: pad.is_pressed(Button::DPadUp),
                    dpad_down: pad.is_pressed(Button::DPadDown),
                    dpad_left: pad.is_pressed(Button::DPadLeft), 
                    dpad_right: pad.is_pressed(Button::DPadRight),
                    start: pad.is_pressed(Button::Start),
                    left_x: pad.value(Axis::LeftStickX),
                    left_y: pad.value(Axis::LeftStickY),
                    east: pad.is_pressed(Button::East),
                    south: pad.is_pressed(Button::South),
                };
            }
            self.world.append_command(SystemCommand::Input(self.input));
        }
        self.gilrs.inc();
        self.world.execute_all_commands();
        self.world.target.execute_all();
    }

    pub fn render(&mut self) {
        self.update();

        let frame = self.swap_chain.acquire_frame(FrameSync::Semaphore(&self.frame_semaphore));
        let view = self.views[frame.id()].clone();
        {
            let mut encoder = self.graphics_pool.acquire_graphics_encoder();

            encoder.clear(&view.0.clone(), CLEAR_COLOR);
            encoder.clear_depth(&view.1.clone(), 1.0);

            self.world.target.render(&view, &mut encoder, &mut self.device);

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
    Move(Vector3<f32>),
}
enum CameraCommand {
    Move(Vector3<f32>),
    LookAt (Point3<f32>),
}
enum SystemCommand {
    Exit,
    Input(GamePad),
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

#[derive(Debug, Copy, Clone, PartialEq)]
struct GamePad {
    dpad_up: bool,
    dpad_down: bool,
    dpad_left: bool,
    dpad_right: bool,
    start: bool,
    south: bool,
    east: bool,
    left_x: f32,
    left_y: f32,
}

impl Default for GamePad {
    fn default() -> GamePad {
        GamePad {
            dpad_up: false,
            dpad_down: false,
            dpad_left: false,
            dpad_right: false,
            start: false,
            left_x: 0.0,
            left_y: 0.0,
            east: false,
            south: false,
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum WorldState {
    Render,
    Pose,
}

struct World<B: gfx::Backend, V> {
    timer: coarsetime::Instant,
    state: WorldState,

    camera: Invoker<CameraCommand, Camera<f32>>,
    avators: Invoker<AvatorCommand, HashMap<i32, GameObject<B, V>>>,
    objects: Vec<GameObject<B, V>>,
    layout: FontLayout<B, FontBundle<B::Resources, pipe_w2::Data<B::Resources>>>,

    sampler: gfx::handle::Sampler<B::Resources>,
    pso_set: PipelineStateObjectSet<B::Resources>,
    inspector: InspectMenu<B>,

    font: Font<B>,
}

struct PipelineStateObjectSet<R: gfx::Resources> {
    pso: gfx::PipelineState<R, pipe_w::Meta>,
    w2: gfx::PipelineState<R, pipe_w2::Meta>,
    p: gfx::PipelineState<R, pipe_p::Meta>,
    pt: gfx::PipelineState<R, pipe_pt::Meta>,
}

fn open_connection() -> Connection {
    Connection::open(&Path::new("file.db")).expect("failed to open sqlite file")
}

impl<B: gfx::Backend> World<B, Vertex> {
    fn new<D: gfx::Device<B::Resources>> (
        device: &mut D,
        aspect: f32,
    ) -> Self {
        use gfx::traits::DeviceExt;

        let conn = open_connection();

        let avators = Invoker::<AvatorCommand, HashMap<i32, GameObject<B, _>>>::new(
            query_entry::<B, D, TextureFormat>(&conn, device, &[1,2]).unwrap()
        );
        let camera = Invoker::<CameraCommand, Camera<f32>>::new(
            Camera::new(
                Point3::new(30.0, -40.0, 30.0),
                Point3::new(0.0, 0.0, 0.0),
                cgmath::PerspectiveFov {
                    fovy: cgmath::Rad(16.0f32.to_radians()),
                    aspect,
                    near: 5.0,
                    far: 1000.0,
            })
        );

        let sampler = {
            let sampler_info = gfx::texture::SamplerInfo::new(
                gfx::texture::FilterMethod::Trilinear,
                gfx::texture::WrapMode::Clamp
            );
            device.create_sampler(sampler_info)
        };
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
            }",
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
            }").expect("failed to build shader");
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
            in vec4 color;
            out vec4 v_Color;
            
            out vec2 v_TexCoord;
            out vec3 _normal;
            
            void main() {
                v_TexCoord = vec2(uv.x, uv.y);
            
                gl_Position = u_model_view_proj * vec4(position, 1.0);
                _normal = normalize(normal);
                v_Color = color;
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
            in vec4 v_Color;
 
            out vec4 Target0;
            
            void main() {
                vec4 texColor = texture(u_texture, v_TexCoord);
            
                float diffuse = clamp(dot(_normal, -u_light), 0.05f, 1.0f);
                vec3 halfLE = normalize(u_eyeDirection);
                float specular = pow(clamp(dot(_normal, halfLE), 0.0, 1.0), 50.0);
                Target0 = vec4(vec3(diffuse) + vec3(specular), texColor.r) + u_ambientColor;
            }").expect("failed to build shader");
            device.create_pipeline_state(
                &shaders,
                gfx::Primitive::TriangleList,
                gfx::state::Rasterizer::new_fill().with_cull_back(),
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

            uniform vec2 u_screen_size;
            uniform vec2 u_cursor;
            
            void main() {
                if (gl_FragCoord.x > u_screen_size.x - 120
                    && gl_FragCoord.x < u_screen_size.x - 40
                    && gl_FragCoord.y > 40
                    && gl_FragCoord.y < 120) {
                    Target0 = vec4(1.0, 1.0, 0.0, 1.0);
                } else {
                    Target0 = v_color;
                }
            }").expect("failed to build shader");
            device.create_pipeline_state(
                &shaders,
                gfx::Primitive::TriangleStrip,
                gfx::state::Rasterizer::new_fill().with_cull_back(),
                pipe_p::new()
            ).expect("failed to create pipeline p")
        };
        let pso_pt = {
            let shaders = device.create_shader_set(b"
            #version 150 core
            
            in vec3 position;
            in vec2 uv;
            in vec4 color;
            out vec2 v_TexCoord;
            out vec4 v_Color;

            uniform vec2 u_screen_size;
            
            void main() {
                vec2 screenOffset = vec2(
                    2 * position.x / u_screen_size.x - 1,
                    2 * position.z / u_screen_size.y - 1
                );
                v_TexCoord = uv;
                gl_Position = vec4(screenOffset, 0.0, 1.0);
                v_Color = color;
            }
            ",
            b"
            #version 150 core

            uniform sampler2D u_texture;
            
            in vec2 v_TexCoord;
            in vec4 v_Color;

            out vec4 Target0;
            
            void main() {
                vec4 texColor = texture(u_texture, v_TexCoord);
                Target0 = vec4(v_Color.rgb, texColor.r * v_Color.a);
            }").expect("failed to build shader");
            device.create_pipeline_state(
                &shaders,
                gfx::Primitive::TriangleList,
                gfx::state::Rasterizer::new_fill().with_cull_back(),
                pipe_pt::new()
            ).expect("failed to create pipeline p")
        };

        let font = {
            let font_chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz0123456789.+-_".chars().map(|c| c).collect();
            Font::from_path(
                "assets/VL-PGothic-Regular.ttf",
                48,
                Some(font_chars.as_slice())
            )
        }.expect("failed to create font");

        let pso_set = PipelineStateObjectSet {
            pso,
            p: pso_p,
            w2: pso_w2,
            pt: pso_pt,
        };
        let objects = Vec::<_>::with_capacity(255);
        let inspector = InspectMenu::new(
            &font,
            [0.03, 0.03, 0.0, 1.0],
        );

        let layout = font.layout::<FontBundle<B::Resources, pipe_w2::Data<B::Resources>>>(
            "".to_string(), [0.0, 0.0], [0.0;4], 0.1
        );

        World {
            avators,
            camera,
            sampler,
            pso_set,
            font,
            inspector,
            timer: coarsetime::Instant::now(),
            state: WorldState::Render,
            objects,
            layout
        }
    }
    fn camera(&self) -> &Camera<f32> {
        &self.camera.target
    }
    fn render<D: gfx::Device<B::Resources>>(&mut self, view: &View<B::Resources>, encoder: &mut gfx::GraphicsEncoder<B>, device: &mut D) {
        let time = self.timer.elapsed().as_f64();

        let (screen_width, screen_height, _, _) = view.0.get_dimensions();
        let screen_size = [screen_width as f32, screen_height as f32];

        if self.state == WorldState::Pose {
            self.inspector.update(view, &self.font, &self.sampler, device, screen_size, "hoge".to_string());
            self.inspector.render(&self.pso_set, encoder);
        }
        {
            self.layout.update(&self.font, view, &self.camera.target, &self.sampler, device, format!("{:?}", &time));
            self.layout.render(&self.pso_set.w2, encoder);
        }


        for obj in self.avators.target.values_mut() {
            obj.update(encoder, view, &self.camera.target, time, &self.sampler);
            obj.render(&self.pso_set.pso, encoder);
        }
    }

    fn execute_all(&mut self) {
        self.avators.execute_all_commands();
        self.camera.execute_all_commands();
    }
}


struct InspectMenu<B: gfx::Backend> {
    color: [f32; 4],
    frame: Option<Bundle<B::Resources, pipe_p::Data<B::Resources>>>,
    text: FontLayout<B, FontBundle<B::Resources, pipe_pt::Data<B::Resources>>>,
}

impl<B: gfx::Backend> InspectMenu<B> {
    fn new(
        font: &Font<B>,
        color: [f32; 4],
    ) -> Self {
       InspectMenu {
            color,
            frame: None,
            text: font.layout::<FontBundle<B::Resources, pipe_pt::Data<B::Resources>>>("".to_string(), [0.0, 0.0], [0.0;4], 0.1)
        }
    }
    fn update<D: gfx::Device<B::Resources>> (
        &mut self,
        view: &View<B::Resources>,
        font: &Font<B>,
        sampler: &gfx::handle::Sampler<B::Resources>,
        device:  &mut D,
        screen_size: [f32; 2],
        text: String,
        ) {
        use gfx::traits::DeviceExt;
        if let None = self.frame {
            let vertex_data = vec!(
                VertexP {
                    position: [-0.95, 0.0, 0.0],
                    color: self.color,
                }, 
                VertexP {
                    position: [0.95, 0.0, 0.0],
                    color: self.color,
                },
                VertexP {
                    position: [-0.95, -0.95, 0.0],
                    color: self.color,
                },
                VertexP {
                    position: [0.95,  -0.95, 0.0],
                    color: self.color,
                },
            );

            let (vbuf, slice) = device.create_vertex_buffer_with_slice(&vertex_data, &[1u32, 0u32, 2u32, 3u32, 1u32][..]);
            let data = pipe_p::Data {
                vbuf,
                out_color: view.0.clone(),
                out_depth: view.1.clone(),
                screen_size,
                cursor: [0.0, 0.0],
            };

            self.frame = Some(
                Bundle {
                    slice,
                    data,
                }
            );
        }
        self.text.update(&font, view, sampler, device, screen_size, text);
    }

    fn render (
        &self,
        pso_set: &PipelineStateObjectSet<B::Resources>,
        encoder: &mut gfx::GraphicsEncoder<B>,
    ) {
        if let Some(ref frame) = self.frame {
            encoder.draw(&frame.slice, &pso_set.p, &frame.data);
        }
        self.text.render(&pso_set.pt, encoder);
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

impl<B: gfx::Backend, V> Command<World<B,V>> for SystemCommand {
    fn get_level(&self) -> Level {
        Level::System
    }
    fn execute(&self, world: &mut World<B,V>) {
        match *self {
            SystemCommand::Input(ref pad) => {
                if pad.dpad_down {
                    world.avators.append_command(AvatorCommand::Move(Vector3::new(0.0,0.5,0.0)))
                }
                if pad.dpad_up {
                    world.avators.append_command(AvatorCommand::Move(Vector3::new(0.0,-0.5,0.0)))
                }
                if pad.dpad_left {
                    world.avators.append_command(AvatorCommand::Move(Vector3::new(-0.5,0.0,0.0)))
                }
                if pad.dpad_right {
                    world.avators.append_command(AvatorCommand::Move(Vector3::new(0.5,0.0,0.0)))
                }
                if pad.start {
                    world.state = if world.state == WorldState::Render {
                        WorldState::Pose
                    } else { 
                        WorldState::Render 
                    };
                }
                world.avators.append_command(AvatorCommand::Move(Vector3::new(pad.left_x, 0.0, 0.0)));
                world.avators.append_command(AvatorCommand::Move(Vector3::new(0.0, pad.left_y, 0.0)));
            },
            SystemCommand::Exit => {
            }
        }
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
            CameraCommand::LookAt(v) => {
                c.look_at(v);
                c.update();
            }
        }
    }
}

impl<B: gfx::Backend, V> Command<GameObject<B, V>> for AvatorCommand {
    fn get_level(&self) -> Level {
        Level::Avator
    }
    fn execute(&self, c: &mut GameObject<B, V>) {
        match *self {
            AvatorCommand::Move(v) => {
                c.translate(v); 
            },
        }
    }
}
impl<B: gfx::Backend, V> Command<HashMap<i32, GameObject<B, V>>> for AvatorCommand {
    fn get_level(&self) -> Level {
        Level::Avator
    }
    fn execute(&self, c: &mut HashMap<i32, GameObject<B, V>>) {
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
    _World,
    System,
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
        color: [f32; 4] = "color",
    }
    pipeline pipe_p {
        vbuf: gfx::VertexBuffer<VertexP> = (),
        out_color: gfx::RenderTarget<ColorFormat> = "Target0",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
        screen_size: gfx::Global<[f32; 2]> = "u_screen_size",
        cursor: gfx::Global<[f32; 2]> = "u_cursor",
    }
    pipeline pipe_pt {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        out_color: gfx::BlendTarget<ColorFormat> = ("Target0", gfx::state::ColorMask::all(), gfx::preset::blend::ALPHA),
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
        u_texture: gfx::TextureSampler<f32> = "u_texture",
        screen_size: gfx::Global<[f32; 2]> = "u_screen_size",
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
        out_color: gfx::BlendTarget<ColorFormat> = ("Target0", gfx::state::ColorMask::all(), gfx::preset::blend::ALPHA),
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
            position,
            target,
            view,
            perspective,
            projection: perspective * view
        }
    }
    fn look_at(&mut self, target: Point3<T>) {
        self.target = target;
    }
    fn direction(& self) -> Vector3<T> {
        self.target - self.position
    }
    fn update(&mut self) {
        self.view = Matrix4::look_at(self.position, self.target, Vector3::new(Zero::zero(), Zero::zero(), One::one()));
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
            joint_weights: [0.0; 4],
            color: [0.0; 4],
        }
    }
}

const CLEAR_COLOR: [f32; 4] = [0.1, 0.2, 0.3, 1.0];

pub struct Entry<R: gfx::Resources, V, View> {
    slice: gfx::Slice<R>,
    vertex_buffer: gfx::handle::Buffer<R, V>,
    texture:  gfx::handle::ShaderResourceView<R, View>
}

fn entry<'e, R, F, V, T>(device: &mut F, vertex_data: &[V], img: &'e Image<T>) -> Entry<R, V, T::View> 
    where 
        R: gfx::Resources,
        F: gfx::Device<R>,
        V: gfx::traits::Pod + gfx::pso::buffer::Structure<gfx::format::Format>,
        T: gfx::format::TextureFormat,
{
    let index_data: Vec<u32> = vertex_data.iter().enumerate().map(|(i, _)| i as u32).collect();
    entry_(device, &vertex_data, &index_data[..], img)
}

fn entry_<'e, R, F, V, T>(device: &mut F, vertex_data: &[V], index_data: &[u32], img: &'e Image<T>) -> Entry<R, V, T::View> 
    where 
        R: gfx::Resources,
        F: gfx::Device<R>,
        V: gfx::traits::Pod + gfx::pso::buffer::Structure<gfx::format::Format>,
        T: gfx::format::TextureFormat,
{
    use gfx::traits::DeviceExt;
    let (vbuf, slice) = device.create_vertex_buffer_with_slice(&vertex_data, index_data);

    let tex_kind = gfx::texture::Kind::D2(img.width, img.height, gfx::texture::AaMode::Single);
    let (_, view) = device.create_texture_immutable_u8::<T>(tex_kind, &[&img.data]).expect("failed to create texture");

    Entry {
        slice,
        vertex_buffer: vbuf,
        texture: view
    }
}




fn query_entry<
    B: gfx::Backend,
    D: gfx::Device<B::Resources>,
    T: gfx::format::TextureFormat>(
    conn: &Connection,
    device: &mut D, 
    ids: &[i32],
) -> RusqliteResult<HashMap<i32, GameObject<B, Vertex>>> {
    use gfx::traits::DeviceExt;

    let mut result = HashMap::default();

    for id in ids {
        let meshes = query_mesh(&conn, id)?;
        let joints = query_skeleton(&conn, id)?;
        let animations = query_animation(&conn, id)?;
        let entries = meshes.iter().map(|&(ref vertex_data, texture_id)| {
            let img = query_texture::<TextureFormat>(&conn, texture_id).expect("failed to create texture");
            entry(device, vertex_data.as_slice(), &img)
        }).collect();

        let skinning_buffer = device.create_constant_buffer(64);

        result.insert(
            id.clone(), 
            GameObject {
                entries,
                position: Point3::new(0.0, 0.0, 0.0),
                // front: Vector3::new(0.0, -1.0, 0.0)
                joints,
                animations,
                skinning_buffer,
                backend: std::marker::PhantomData::<B>,
                bundle: None,
            }
        );
    }
    Ok(result)
}


struct GameObject<B: gfx::Backend, V> {
    entries: Vec<Entry<B::Resources, V, [f32;4]>>,
    position: Point3<f32>,
    // front: Vector3<f32>,
    joints: Vec<Joint>,
    animations: Vec<Vec<(f32, Animation)>>,

    skinning_buffer: gfx::handle::Buffer<B::Resources, Skinning>,

    backend: std::marker::PhantomData<B>,
    bundle: Option<Vec<Bundle<B::Resources, pipe_w::Data<B::Resources>>>>,
}

trait Translate<T: cgmath::BaseFloat> {
    fn translate(&mut self, v: Vector3<T>);
}

impl<B: gfx::Backend, V> Translate<f32> for GameObject<B, V>
{
    fn translate(&mut self, v: Vector3<f32>) {
        self.position += v;
    }
}
impl<T: cgmath::BaseFloat> Translate<T> for Camera<T> {
    fn translate(&mut self, v: Vector3<T>) {
        self.position += v;
    }
}

trait GraphicsComponent<
    B: gfx::Backend,
    D: gfx::Device<B::Resources>,
> {
    fn render(
        &self,
        view: &View<B::Resources>,
        pso: &PipelineStateObjectSet<B::Resources>,
        encoder: &mut gfx::GraphicsEncoder<B>,
        sampler: &gfx::handle::Sampler<B::Resources>,
        device: &mut D, 
        screen_size: [f32; 2],
    );
}


impl<B: gfx::Backend> GameObject<B, Vertex> {
    fn update(
        &mut self,
        encoder: &mut gfx::GraphicsEncoder<B>,
        view: &View<B::Resources>,
        camera: &Camera<f32>,
        elapsed: f64,
        sampler: &gfx::handle::Sampler<B::Resources>,
    ) {
        let mv = camera.view * Matrix4::from_translation(self.position.to_vec());
        let mvp = camera.perspective * mv;
        {
            let a = self.get_skinning(elapsed);
            encoder.update_buffer(&self.skinning_buffer, &a, 0).expect("ub");
        }
        let bundle = self.entries.iter().map(|entry| {
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
            Bundle {
                slice: entry.slice.clone(),
                data,
            }
        }).collect();
        self.bundle = Some(bundle);
    }
    fn render(
        &self,
        pso: &gfx::PipelineState<B::Resources, pipe_w::Meta>,
        encoder: &mut gfx::GraphicsEncoder<B>,
    ) {
        if let Some(ref bundle) = self.bundle {
            for entry in bundle {
                encoder.draw(&entry.slice, pso, &entry.data);
            }
        }
    }
}

struct Bundle<R: gfx::Resources, D: gfx::pso::PipelineData<R>> {
    slice: gfx::Slice<R>,
    data: D,
}

struct FontBundle<R: gfx::Resources, D: gfx::pso::PipelineData<R>> {
    slice: gfx::Slice<R>,
    data: D,
    vertex_data: Vec<Vertex>,
    index_data: Vec<u32>,
}

fn font_entry<B: gfx::Backend, R: gfx::Resources, D: gfx::Device<R>>(
    device: &mut D,
    font: &Font<B>,
    vertex_data: &mut Vec<Vertex>,
    index_data: &mut Vec<u32>,
    text: &str,
    pos: [f32;2],
    color: [f32;4],
    scale: f32) -> Entry<R, Vertex, f32> 
{
    vertex_data.clear();
    index_data.clear();

    let (mut x, z, mut y) = (pos[0], 0.0, pos[1]);

    let mut min_y_end = y as i32;
    for l in text.split('\n') {
        for ch in l.chars() {
            let ch_info = match font.chars.get(&ch) {
                Some(info) => info,
                None => continue,
            };
            let x_offset = (x + ch_info.x_offset as f32) * scale;
            let y_offset = (y - ch_info.y_offset as f32) * scale;
            let tex = ch_info.tex;
            let x_end = x_offset + ch_info.width as f32 * scale;
            let y_end = y_offset - ch_info.height as f32 * scale;
            min_y_end = std::cmp::min(min_y_end, y_end as i32);

            let index = vertex_data.len() as u32;

            vertex_data.push(
                Vertex { 
                    position: [x_offset, z, y_offset],
                    normal: [0.0, 1.0, 0.0],
                    uv: [tex[0], tex[1]] ,
                    joint_indices: [0;4], joint_weights: [0.0;4], color 
                }
            );
            vertex_data.push(
                Vertex { 
                    position: [x_offset, z, y_end],
                    normal: [0.0, 1.0, 0.0],
                    uv: [tex[0], tex[1] + ch_info.tex_height], 
                    joint_indices: [0;4], joint_weights: [0.0;4], color
                }
            );
            vertex_data.push(
                Vertex { 
                    position: [x_end, z, y_end],
                    normal: [0.0, 1.0, 0.0],
                    uv: [tex[0] + ch_info.tex_width, tex[1] + ch_info.tex_height], 
                    joint_indices: [0;4], joint_weights: [0.0;4], color
                }
            );
            vertex_data.push(
                Vertex { 
                    position: [x_end, z, y_offset],
                    normal: [0.0, 1.0, 0.0],
                    uv: [tex[0] + ch_info.tex_width, tex[1]] ,
                    joint_indices: [0;4], joint_weights: [0.0;4], color
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
        x = pos[0];
        y = min_y_end as f32;
        min_y_end = pos[1] as i32;
    }
    entry_(
        device,
        &vertex_data,
        &index_data,
        &font.texture,
    )
}


impl<B: gfx::Backend> FontLayout<B, FontBundle<B::Resources, pipe_pt::Data<B::Resources>>> {
    fn should_update(&self, text: &str) -> bool {
        self.text != text
    }
    fn update<D: gfx::Device<B::Resources>>(
        &mut self,
        font: &Font<B>,
        view: &View<B::Resources>,
        sampler: &gfx::handle::Sampler<B::Resources>,
        device:  &mut D,
        screen_size: [f32; 2],
        text: String,
    ) {
        if self.should_update(&text) {
            match self.data {
                Some(FontBundle {
                    ref mut data,
                    ref mut slice,
                    ref mut vertex_data,
                    ref mut index_data,
                }) => {
                    let font_entry = font_entry(device, font, vertex_data, index_data, &text, self.position, self.color, self.scale);
                    *data = pipe_pt::Data {
                        vbuf: font_entry.vertex_buffer,
                        u_texture: (font_entry.texture, sampler.clone()),
                        out_color: view.0.clone(),
                        out_depth: view.1.clone(),
                        screen_size
                    };
                    *slice = font_entry.slice;
                },
                None => {
                    println!("assigned");
                    
                    let mut vertex_data = Vec::with_capacity(255);
                    let mut index_data = Vec::with_capacity(255);

                    let font_entry = font_entry(device, font, &mut vertex_data, &mut index_data, &text, self.position, self.color, self.scale);
                    let data = pipe_pt::Data {
                        vbuf: font_entry.vertex_buffer,
                        u_texture: (font_entry.texture, sampler.clone()),
                        out_color: view.0.clone(),
                        out_depth: view.1.clone(),
                        screen_size
                    };

                    self.data = Some(FontBundle {
                        data, 
                        slice: font_entry.slice,
                        index_data,
                        vertex_data
                    });
                }
            };
        }
        self.text = text;
    }

    fn render(
        &self,
        pso: &gfx::PipelineState<B::Resources, pipe_pt::Meta>,
        encoder: &mut gfx::GraphicsEncoder<B>,
    ) {
        if let Some(ref bundle) = self.data {
            encoder.draw(&bundle.slice, pso, &bundle.data);
        }
    }
}

impl<B: gfx::Backend> FontLayout<B, FontBundle<B::Resources, pipe_w2::Data<B::Resources>>> {
    fn should_update(&self, text: &str) -> bool {
        self.text != text
    }
    fn update<D: gfx::Device<B::Resources>>(
        &mut self,
        font: &Font<B>,
        view: &View<B::Resources>,
        camera: &Camera<f32>,
        sampler: &gfx::handle::Sampler<B::Resources>,
        device:  &mut D, 
        text: String,
    ) {

        if self.should_update(&text) {
            match self.data {
                Some(FontBundle {
                    ref mut data,
                    ref mut slice,
                    ref mut vertex_data,
                    ref mut index_data,
                    ..
                }) => {

                    let font_entry = font_entry(device, font, vertex_data, index_data, &text, self.position, self.color, self.scale);

                    *data = pipe_w2::Data {
                        vbuf: font_entry.vertex_buffer,
                        u_model_view_proj: camera.projection.into(),
                        u_model_view: camera.view.into(),
                        u_light: [1.0, 0.5, -0.5f32],
                        u_ambient_color: [0.00, 0.00, 0.01, 0.4],
                        u_eye_direction: camera.direction().into(),
                        u_texture: (font_entry.texture, sampler.clone()),
                        out_color: view.0.clone(),
                        out_depth: view.1.clone()
                    };
                    *slice = font_entry.slice;
               }, 
                None => {
                    let mut vertex_data = Vec::with_capacity(255);
                    let mut index_data = Vec::with_capacity(255);
                    let font_entry = font_entry(device, font, &mut vertex_data, &mut index_data, &text, self.position, self.color, self.scale);

                    let data = pipe_w2::Data {
                        vbuf: font_entry.vertex_buffer,
                        u_model_view_proj: camera.projection.into(),
                        u_model_view: camera.view.into(),
                        u_light: [1.0, 0.5, -0.5f32],
                        u_ambient_color: [0.00, 0.00, 0.01, 0.4],
                        u_eye_direction: camera.direction().into(),
                        u_texture: (font_entry.texture, sampler.clone()),
                        out_color: view.0.clone(),
                        out_depth: view.1.clone()
                    };
                    self.data = Some(FontBundle {
                        data, 
                        slice: font_entry.slice,
                        vertex_data,
                        index_data
                    });

                }
            }
        }
    }

    fn render(
        &self,
        pso: &gfx::PipelineState<B::Resources, pipe_w2::Meta>,
        encoder: &mut gfx::GraphicsEncoder<B>,
        ) {
        if let Some(ref bundle) = self.data {
            encoder.draw(&bundle.slice, pso, &bundle.data);
        }

    }
}

impl<B: gfx::Backend, V> GameObject<B, V> {
    fn get_skinning(&self, time: f64) -> Vec<Skinning> {
        if self.joints.len() > 0 {
            let mut local = Vec::<Matrix4<f32>>::with_capacity(255);
            self.joints.iter().map(|j| {

                let p = match local.get(j.parent as usize) {
                    Some(p) => { *p },
                    _ => {
                        cgmath::One::one()
                    }
                };
           
                match self.animations.get(j.joint_index as usize) {
                    Some(v) => {
                        let length = v.len();

                        let transform = (
                            p * if length > 0 {
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
                            }
                        ).into();

                        Skinning{ 
                            transform,
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
    fn _get_skinning_at(&self, index: usize) -> Vec<Skinning> {
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
                               r.get::<&str,f64>("JointWeight4") as f32],
              color: [0.0;4]
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

fn query_texture<T>(conn: &Connection, texture_id: i32) -> RusqliteResult<Image<T>> 
    where 
        T: gfx::format::TextureFormat
{
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
            format: std::marker::PhantomData::<T>
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
        let (joint_index, parent, bind, inverse) = r?;

        let joint = Joint {
            joint_index,
            global: bind,
            bind,
            parent,
            inverse,
        };

        joints.insert(joint_index as usize, joint);
    }
    Ok(joints)
}

