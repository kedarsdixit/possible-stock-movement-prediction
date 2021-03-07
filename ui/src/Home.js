import React, { Component } from "react";
import "./App.css";
import Form from "react-bootstrap/Form";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Button from "react-bootstrap/Button";
import "bootstrap/dist/css/bootstrap.css";
import Graph from "./Graph";
import { Table } from "semantic-ui-react";

class Home extends Component {
  constructor(props) {
    super(props);

    this.state = {
      isLoading: false,
      formData: {
        textfield1: "",
      },
      result: [],
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

  handlePredictClick = (event) => {
    const formData = this.state.formData;
    this.setState({ isLoading: true });
    fetch("http://127.0.0.1:5000/prediction/", {
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "POST",
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((response) => {
        this.setState({
          result: response.result,
          isLoading: false,
        });
      });
    if (this.state.result.length > 0) console.log(this.state.result);
  };

  handleCancelClick = (event) => {
    this.setState({ result: [] });
  };

 
  render() {
    const isLoading = this.state.isLoading;
    const formData = this.state.formData;
    const result = this.state.result;

    return (
      <Container>
        <div className="content">
          <Form>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>Company Name </Form.Label>
                <Form.Control
                  as="select"
                 s name="textfield1"
                  value={formData.textfield1}
                  onChange={this.handleChange}
                >
                  <option default>
                    Search for stocks from the suggestions
                  </option>
                  <option>WIPRO</option>
                  <option>TATAMOTORS</option>
                  <option>ONGC</option>
                  <option>CIPLA</option>
                  <option>SBIN</option>
                </Form.Control>
              </Form.Group>
            </Form.Row>

            <Row>
              <Col>
                <Button
                  block
                  variant="success"
                  disabled={isLoading}
                  onClick={!isLoading ? this.handlePredictClick : null}
                >
                  {isLoading ? "Making prediction" : "Predict"}
                </Button>
              </Col>
              <Col>
                <Button
                  block
                  variant="danger"
                  disabled={isLoading}
                  onClick={this.handleCancelClick}
                >
                  Reset prediction
                </Button>
              </Col>
            </Row>
          </Form>
          {this.state.result.length == 0 ? null : (
            <Row>
              <Col classname="result-container">
                {"\n"}
                {"\n"}
                <h4>
                  {this.state.formData.textfield1} Company Prediction for next
                  five days from today :
                </h4>
                <Table>
                  <Table.Header>
                    <Table.Row>
                      <Table.HeaderCell>Day</Table.HeaderCell>
                      <Table.HeaderCell>Date</Table.HeaderCell>
                      <Table.HeaderCell>Closing Price</Table.HeaderCell>
                    </Table.Row>
                  </Table.Header>
                  <Table.Body>
                    {JSON.parse(this.state.result).map((close, index) => (
                      <Table.Row key={index}>
                        <Table.Cell>{index + 1}</Table.Cell>
                        <Table.Cell>{close.Date}</Table.Cell>
                        <Table.Cell>{close.close}</Table.Cell>
                      </Table.Row>
                    ))}
                  </Table.Body>
                </Table>
              </Col>
              <Graph formData={this.state.formData.textfield1}></Graph>;
            </Row>
          )}
        </div>
      </Container>
    );
  }
}

export default Home;
