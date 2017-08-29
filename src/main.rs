extern crate gfx;
extern crate glutin;
extern crate gfx_window_glutin;
extern crate gfx_device_gl;

extern crate parti_game as game;


pub fn main() {
    use gfx::Device;

    let width = 1024;
    let height = 768;

    let (window, mut device, factory, main_color, main_depth) = {
        let builder = glutin::WindowBuilder::new()
            .with_title("PARTI")
            .with_dimensions(width, height)
            .with_vsync();
        gfx_window_glutin::init::<game::ColorFormat, game::DepthFormat>(builder)
    };
    let mut app = game::App::new(
        factory, main_color, main_depth, width, height
    );

    'main: loop {
        
        for event in window.poll_events() {
            match event {
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) 
                  => break 'main, 
                _ => app.handle_input(event) 
            }
        }
        app.render(&mut device);

        let _ = window.swap_buffers();
        device.cleanup();
    }
}

