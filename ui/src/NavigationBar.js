import React from "react";
import { Nav, Navbar } from "react-bootstrap";
import logo from "./images/stocks_img.png";
import styled from "styled-components";
const Styles = styled.div`
  .navbar {
    background-color: #e8b13d;
  }
  a,
  .navbar-nav,
  .navbar-light .nav-link {
    color: #101010;
    &:hover {
      color: white;
    }
  }
  .navbar-brand {
    font-size: 1.4em;
    color: #101010;
    &:hover {
      color: white;
    }
  }
  .form-center {
    position: absolute !important;
    left: 25%;
    right: 25%;
  }
`;
export const NavigationBar = () => (
  <Styles>
    <Navbar expand="lg">
      <Navbar.Brand href="/">
        <img src={logo} width="100px" height="50px" />
      </Navbar.Brand>
      <Navbar.Toggle aria-controls="basic-navbar-nav" />

      <Navbar.Collapse id="basic-navbar-nav">
        <Nav className="ml-auto">
          <Nav.Item>
            <Nav.Link href="/">
              <b>Home</b>
            </Nav.Link>
          </Nav.Item>
          <Nav.Item>
            <Nav.Link href="/prediction">
              <b>Graph</b>
            </Nav.Link>
          </Nav.Item>
          <Nav.Item>
            <Nav.Link href="/about">
              <b>About</b>
            </Nav.Link>
          </Nav.Item>
        </Nav>
      </Navbar.Collapse>
    </Navbar>
  </Styles>
);
