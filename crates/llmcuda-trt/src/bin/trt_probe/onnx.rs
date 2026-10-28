//! Just enough ONNX protobuf to hand TensorRT's parser a graph.
//!
//! Hand-encoded wire format for the handful of messages U4 and U5 need. The
//! trunk's emitter (`llmcuda-onnx`) generates its types from `onnx.proto3`;
//! this exists only so the toolchain probe does not depend on that crate.
//! Field numbers are from `onnx/onnx.proto3`.

/// `TensorProto.DataType`.
#[derive(Clone, Copy)]
pub enum Elem {
    Float = 1,
    Float16 = 10,
}

/// Opset of the default domain these graphs import. MatMul and Cast are
/// unchanged across the range TensorRT 11 accepts.
pub const OPSET: i64 = 17;
pub const IR_VERSION: i64 = 8;

#[derive(Default)]
struct Msg(Vec<u8>);

impl Msg {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }
    fn tag(&mut self, field: u32, wire: u32) {
        self.varint(u64::from(field << 3 | wire));
    }
    fn int(&mut self, field: u32, v: i64) -> &mut Self {
        self.tag(field, 0);
        self.varint(v as u64);
        self
    }
    fn bytes(&mut self, field: u32, b: &[u8]) -> &mut Self {
        self.tag(field, 2);
        self.varint(b.len() as u64);
        self.0.extend_from_slice(b);
        self
    }
    fn str(&mut self, field: u32, s: &str) -> &mut Self {
        self.bytes(field, s.as_bytes())
    }
    fn msg(&mut self, field: u32, m: &Msg) -> &mut Self {
        self.bytes(field, &m.0)
    }
}

/// `ValueInfoProto` for a tensor of fixed shape.
fn value_info(name: &str, elem: Elem, dims: &[i64]) -> Msg {
    let mut shape = Msg::default();
    for &d in dims {
        let mut dim = Msg::default();
        dim.int(1, d); // Dimension.dim_value
        shape.msg(1, &dim); // TensorShapeProto.dim
    }
    let mut tensor = Msg::default();
    tensor.int(1, elem as i64).msg(2, &shape); // elem_type, shape
    let mut ty = Msg::default();
    ty.msg(1, &tensor); // TypeProto.tensor_type
    let mut vi = Msg::default();
    vi.str(1, name).msg(2, &ty);
    vi
}

pub enum Attr<'a> {
    Int(i64),
    Str(&'a str),
}

pub struct Node<'a> {
    pub op_type: &'a str,
    pub domain: &'a str,
    pub inputs: &'a [&'a str],
    pub outputs: &'a [&'a str],
    pub attrs: &'a [(&'a str, Attr<'a>)],
}

pub struct Initializer<'a> {
    pub name: &'a str,
    pub elem: Elem,
    pub dims: &'a [i64],
    pub raw: &'a [u8],
}

pub struct Graph<'a> {
    pub nodes: &'a [Node<'a>],
    pub inputs: &'a [(&'a str, Elem, &'a [i64])],
    pub outputs: &'a [(&'a str, Elem, &'a [i64])],
    pub initializers: &'a [Initializer<'a>],
    /// Custom domains to import at version 1.
    pub custom_domains: &'a [&'a str],
}

/// Serialize a `ModelProto`.
pub fn model(g: &Graph<'_>) -> Vec<u8> {
    let mut graph = Msg::default();
    for (i, n) in g.nodes.iter().enumerate() {
        let mut node = Msg::default();
        for input in n.inputs {
            node.str(1, input);
        }
        for output in n.outputs {
            node.str(2, output);
        }
        node.str(3, &format!("{}_{i}", n.op_type)).str(4, n.op_type);
        for (name, value) in n.attrs {
            let mut a = Msg::default();
            a.str(1, name);
            match value {
                Attr::Int(v) => a.int(3, *v).int(20, 2), // i, type INT
                Attr::Str(s) => a.str(4, s).int(20, 3),  // s, type STRING
            };
            node.msg(5, &a);
        }
        if !n.domain.is_empty() {
            node.str(7, n.domain);
        }
        graph.msg(1, &node);
    }
    graph.str(2, "trt_probe");
    for init in g.initializers {
        let mut t = Msg::default();
        for &d in init.dims {
            t.int(1, d);
        }
        t.int(2, init.elem as i64)
            .str(8, init.name)
            .bytes(9, init.raw);
        graph.msg(5, &t);
    }
    for (name, elem, dims) in g.inputs {
        graph.msg(11, &value_info(name, *elem, dims));
    }
    for (name, elem, dims) in g.outputs {
        graph.msg(12, &value_info(name, *elem, dims));
    }

    let mut m = Msg::default();
    m.int(1, IR_VERSION).str(2, "llmcuda-trt trt_probe");
    let mut opset = Msg::default();
    opset.str(1, "").int(2, OPSET);
    m.msg(8, &opset);
    for domain in g.custom_domains {
        let mut o = Msg::default();
        o.str(1, domain).int(2, 1);
        m.msg(8, &o);
    }
    m.msg(7, &graph);
    m.0
}
