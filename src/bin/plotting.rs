use plotters::{
    prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea},
    series::LineSeries,
    style::{RED, WHITE},
};

static PLOTS_FOLDER: &str = "/Users/ojoseph/Desktop/personal/raw_code_samples/plots/";

fn main() -> () {
    let destination = format!("{}example-mesh.png", PLOTS_FOLDER);
    let drawing_area = BitMapBackend::new(&destination, (1024, 768)).into_drawing_area();

    drawing_area.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&drawing_area)
        .build_cartesian_2d(-3.14..3.14, -1.2..1.2)
        .unwrap();
    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(
            (-314..314).map(|x| x as f64 / 100.0).map(|x| (x, x.sin())),
            &RED,
        ))
        .unwrap();
}
