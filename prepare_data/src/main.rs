#[macro_use]
extern crate lazy_static;
extern crate quick_xml;
extern crate rand;
extern crate regex;

use std::ascii::AsciiExt;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::str;
use std::path::Path;

use quick_xml::Reader;
use quick_xml::events::Event;
use rand::prelude::*;
use regex::Regex;


#[derive(Copy, Clone)]
enum ParserState {
	Initial,
	Text,
	Other
}


fn main() {
	let filename_in = env::args().nth(1).expect("Error: no input filename given");
	let filename_out = env::args().nth(2).expect("Error: no output filename given");
    let source = Path::new(&filename_in);
	let mut reader = Reader::from_file(source).expect("Cannot read source file");
	reader.trim_text(true);

	let mut words = Vec::new();
	let mut buf = Vec::new();

	let mut state = ParserState::Initial;

	loop {
	    match (state, reader.read_event(&mut buf)) {
	        (_, Ok(Event::Start(ref e))) => {
	        	if e.name() == b"text" {
	        		state = ParserState::Text;
	        	}
	        },
	        (_, Ok(Event::End(..))) => {
	        	state = ParserState::Other;
	        },
	        (ParserState::Text, Ok(Event::Text(e))) => {
	        	let text = e.unescape_and_decode(&reader).unwrap();
	        	let mut lines = text.lines();
	        	let header = lines.next().unwrap();

	        	if !header.contains("({{Sprache|Deutsch}}") {
	        		continue;
	        	}

	        	let hyphenation_pos = lines.position(|l| l.contains("{{Worttrennung}}"));

	        	if hyphenation_pos.is_some() {
	        		let word = lines.nth(0).unwrap();
	        		words.extend(process_word(word));

	        		if words.len() % 1000 == 0 {
	        			println!("Processed {} words", words.len());
	        		}
	        	}
	        },
	        (_, Ok(Event::Eof)) => break,
	        (_, Err(e)) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
	        _ => (),
	    }

	    buf.clear();
	}

	// Randomize line order
	println!("Randomizing line order...");
	let mut rng = rand::thread_rng();
	let words = words.as_mut_slice();
	words.shuffle(&mut rng);

	// Write output file
	println!("Writing output file...");
	let mut output = File::create(filename_out).expect("Cannot create output file");
	for word in words {
		writeln!(output, "{}", word).unwrap();
	}

	println!("Done!");
}

fn process_word(word: &str) -> Vec<String> {
	lazy_static! {
		static ref RE_STRIP_JUNK: Regex = Regex::new(r"\{\{.*\}\}|<.*>|''\)").unwrap();
		static ref RE_ENDING: Regex = Regex::new(r"\((\w)\)\z").unwrap();
	}

	let split_at: &[char] = &[',', ':'];
	word.split(split_at)
		.map(|word| {
			word.trim()
				.to_owned()
		})
		.filter(|word| {
			word.chars().count() >= 3
		})
		.filter(|word| {
			!word.starts_with("''")
				&& !word.starts_with('-')
				&& !word.starts_with('(')
				&& !word.starts_with(')')
				&& !word.starts_with('·')
				&& !word.starts_with("//www.duden.de")
				&& !word.ends_with('-')
				&& !word.ends_with('.')
				&& !word.ends_with('(')
				&& !word.ends_with('·')
				&& !word.contains("··")
				&& !word.contains(' ')
				&& !word.contains('&')
				&& !word.contains('.')
				&& !word.contains('\'')
				&& !word.contains('/')
				&& !word.contains('!')
				&& !word.contains('+')
				&& !word.contains('%')
				&& !word.contains('{')
				&& !word.contains('}')
				&& !word.contains('[')
				&& !word.contains(']')
				&& !word.contains('@')
				&& word.chars().all(|c| {
					char::is_ascii(&c)
						|| c == '·'
						|| c == 'ä' || c == 'ö' || c == 'ü'
						|| c == 'Ä' || c == 'Ö' || c == 'Ü'
						|| c == 'ß'
				})
		})
		.map(|word| {
			RE_STRIP_JUNK.replace(&word, "").into_owned()
		})
		.filter(|word| {
			!word.is_empty()
				&& word != "()"
				&& word != "</ref>"
				&& word != "Ærø"
				&& word != "CO₂"
				&& word != "···"
		})
		.flat_map(|word| {
			if let Some(m) = RE_ENDING.captures(&word) {
				let c = m.get(1).unwrap().as_str().to_owned();

				let word = word.replace(&format!("({})", c), "");
				vec![word.clone(), word + &c]
			} else {
				vec![word.clone()]
			}
		})
		.filter(|word| {
			!word.ends_with(')')
				&& word.chars().nth(1) != Some('·')
				&& word.chars().count() >= 3
				&& !word.chars().all(|c| c.is_digit(10))
		})
		.collect()
}