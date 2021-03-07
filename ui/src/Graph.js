import React from "react";
import Chart from "./Chart";
import "./App.css";
import { csvParse } from "d3-dsv";
import { timeParse } from "d3-time-format";

export default class Graph extends React.Component {
  constructor(props) {
    super(props);
  }

  callingFunction = (event) => {
    this.getData().then((data) => {
      this.setState({ data });
    });
  };
  parseData(parse) {
    return function (d) {
      d.date = parse(d.Date);
      d.open = +d.Open;
      d.high = +d.High;
      d.low = +d.Low;
      d.close = +d.Close;
      d.volume = +d.Volume;

      return d;
    };
  }
  getData() {
    const companyname = this.props.formData;
    const parseDate = timeParse("%Y-%m-%d");
    const promiseMSFT = fetch(`${companyname}.csv`)
      .then((response) => response.text())
      .then((data) => csvParse(data, this.parseData(parseDate)));
    return promiseMSFT;
  }

  render() {
    if (this.props.formData != "") {
      {
        this.callingFunction();
      }
    }
    if (this.state == null) {
      return <div>...</div>;
    }
    return (
      <div className="content">
        <Chart data={this.state.data} />
      </div>
    );
  }
}

