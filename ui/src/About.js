import React, { Component } from "react";
import "./App.css";
import Container from "react-bootstrap/Container";

class About extends Component {
  constructor(props) {
    super(props);

    this.state = {
      isLoading: false,
    };
  }

  render() {
    return (
      <Container>
        <div className="content">
          <h3>Stock Market Prediction</h3>
          <p>
            This Web App predicts the closing price of selected company for the
            next 5 days. Also it displays the Technical Indicators using Graphs
            for selected company.
          </p>
        </div>
      </Container>
    );
  }
}

export default About;
