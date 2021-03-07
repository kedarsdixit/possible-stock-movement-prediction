import React, { Component } from "react";
import "./App.css";
import Form from "react-bootstrap/Form";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Button from "react-bootstrap/Button";
import "bootstrap/dist/css/bootstrap.css";
import ChartComponent from "./Prediction";
import TI from "./TI";

class Prediction extends Component {
  constructor(props) {
    super(props);

    this.state = {
      isLoading: false,
      formData: {
        textfield1: "",
        textfield2: "",
      },
    };
  }

  handleChange = (event) => {
    const value = event.target.value;
    const name = event.target.name;
    var formData = this.state.formData;
    formData[name] = value;
    this.setState({
      formData,
    });
  };

  handleClick = (event) => {
    const formData = this.state.formData;
    this.setState({ isLoading: true });
    this.setState({
      isLoading: false,
    });
    console.log(this.state);
  };

  render() {
    const isLoading = this.state.isLoading;
    const formData = this.state.formData;
    
    return (
      <Container>
        <div className="content">
          <Form>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>Company Name </Form.Label>
                <Form.Control
                  as="select"
                  name="textfield1"
                  value={formData.textfield1}
                  onChange={this.handleChange}
                >
                  <option default>
                    Search for company from the suggestions
                  </option>
                  <option>WIPRO</option>
                  <option>TATAMOTORS</option>
                  <option>ONGC</option>
                  <option>CIPLA</option>
                  <option>SBIN</option>
                </Form.Control>
              </Form.Group>

              <Form.Group as={Col}>
                <Form.Label>Technical Indicator </Form.Label>
                <Form.Control
                  as="select"
                  name="textfield2"
                  value={formData.textfield2}
                  onChange={this.handleChange}
                >
                  <option default>
                    Search for Technical Indicator from the suggestions
                  </option>
                  <option>RSI</option>
                  <option>MACD</option>
                  <option>Moving Average</option>
                  <option>Stochastics</option>
                  <option>Bollinger Bands</option>
                </Form.Control>
              </Form.Group>
            </Form.Row>

            
          </Form>
          <TI
            txt1={this.state.formData.textfield1}
            txt2={this.state.formData.textfield2}
          ></TI>
        </div>
      </Container>
    );
  }
}

export default Prediction;
